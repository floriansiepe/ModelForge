import os
import shutil
from logging import Logger
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from distributed import Client, LocalCluster
from dotenv import find_dotenv, load_dotenv

from modelforge.datasets.weather_models.scripts.weather_model_trainer import (
    WeatherModelTrainer,
    WeatherProbModelTrainer,
)
from modelforge.shared.logger import logger_factory


@click.command()
@click.argument("raw_filepath", type=click.Path(exists=True))
@click.argument("processed_filepath", type=click.Path())
@click.option("--model_type", default="xgboost")
def main(raw_filepath: str, processed_filepath: str, model_type: str):
    logger = logger_factory(__name__)
    logger.info("Clearing previous point and models")
    shutil.rmtree(f"{processed_filepath}", ignore_errors=True)
    os.makedirs(f"{processed_filepath}/point", exist_ok=True)

    # Ignore point before 2015, as e.g. no ev value is reported
    logger.info("Reading point files")
    data = pd.read_feather(f"{raw_filepath}/data_RL18.feather").drop(
        columns=["lon", "lat", "alt"]
    )
    data.station = pd.to_numeric(data.station, downcast="integer")

    # drop soil moisture predictions due to missing values
    # Note that this is a minor change compared to the paper, but does not have a significant effect
    data = data.drop(["sm_mean", "sm_var"], axis=1)

    # split into train and test point
    eval_start = 1626724
    train_end = 1626723

    train = data.iloc[:train_end]
    test = data.iloc[eval_start:]

    entity_ids = train["station"].unique().tolist()
    dfs_train = []
    dfs_test = []
    for entity_id in entity_ids:
        dfs_train.append(train[train["station"] == entity_id])
        dfs_test.append(test[test["station"] == entity_id])

    client = LocalCluster().get_client()

    trained_entities_prob, global_error_prob = prepare_data_set_and_train_prob(
        processed_filepath, client, entity_ids, dfs_train, dfs_test, logger
    )

    errors_prob = [
        entity.metadata["model_metrics"]["test_crps"]
        for entity in trained_entities_prob
    ]
    error = np.mean(errors_prob)
    plt.hist(errors_prob)
    plt.axvline(global_error_prob, color="r", linestyle="--")
    plt.savefig(f"{raw_filepath}/error_hist_prob.png")
    plt.close()
    logger.info(
        f"Trained {len(trained_entities_prob)} probabilistic models. Mean CRPS is {error}"
    )

    client.close()


def prepare_data_set_and_train_prob(
    processed_filepath: str,
    client: Client,
    entity_ids: list[str],
    dfs_train: list[pd.DataFrame],
    dfs_test: list[pd.DataFrame],
    logger: Logger,
):
    prob_trainer = WeatherProbModelTrainer(logger, processed_filepath + "/prob", "xgb")
    global_model = prob_trainer.train(
        "global", pd.concat(dfs_train), pd.concat(dfs_test)
    )
    global_error = global_model.metadata["model_metrics"]["test_crps"]
    shutil.rmtree(processed_filepath + "/prob/point/global", ignore_errors=True)

    weather_entities = client.map(prob_trainer.train, entity_ids, dfs_train, dfs_test)
    logger.info(f"Submitted {len(weather_entities)} training tasks")
    weather_entities = client.gather(weather_entities)
    return [entity for entity in weather_entities if entity is not None], global_error


def prepare_data_set_and_train_reg(
    processed_filepath: str,
    client: Client,
    entity_ids: list[str],
    dfs_train: list[pd.DataFrame],
    dfs_test: list[pd.DataFrame],
    logger: Logger,
    model_type: str,
):
    reg_trainer = WeatherModelTrainer(logger, processed_filepath + "/reg", model_type)
    global_model = reg_trainer.train(
        "global", pd.concat(dfs_train), pd.concat(dfs_test)
    )
    global_error = global_model.metadata["model_metrics"]["test_mean_absolute_error"]
    shutil.rmtree(processed_filepath + "/reg/point/global", ignore_errors=True)
    logger.info("Trained global set. Error: " + str(global_error))

    if model_type == "nn":
        weather_entities = []
        for entity_id, df_train, df_test in zip(entity_ids, dfs_train, dfs_test):
            weather_entities.append(reg_trainer.train(entity_id, df_train, df_test))
    else:
        weather_entities = client.map(
            reg_trainer.train, entity_ids, dfs_train, dfs_test
        )
        logger.info(f"Submitted {len(weather_entities)} training tasks")
        weather_entities = client.gather(weather_entities)
    return [entity for entity in weather_entities if entity is not None], global_error


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
