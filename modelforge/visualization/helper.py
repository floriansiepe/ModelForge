import json
import os
import re

import pandas as pd

from modelforge.datasets.datasets import (
    anomaly_dataset,
    heating_dataset,
    immoscout_house_price_dataset,
    weather_dataset_probabilistic_regression,
)

datasets = ["anomaly", "heating", "house_price", "weather_probabilistic"]
dataset_sizes = [
    anomaly_dataset("score", "../../data/processed/anomaly_models").size,
    heating_dataset("../../data/processed/heating_models/data").size,
    immoscout_house_price_dataset(
        "../../data/processed/immoscout24_models/house_price/data"
    ).size,
    weather_dataset_probabilistic_regression(
        "../../data/processed/weather_models/prob/data"
    ).size,
]
losses = ["roc_auc", "mae", "mape", "crps"]
losses_maximize = [True, False, False, False]


def clean_pipeline_group(pipeline: str) -> str:
    if "pairwise_loss" in pipeline:
        return "prediction_loss"
    if "prediction_loss" in pipeline:
        return "prediction_loss"
    if "prediction" in pipeline:
        return "prediction"
    if "prediction_set" in pipeline:
        return "prediction"
    if "shap" in pipeline:
        return "shap"
    if "shap_set" in pipeline:
        return "shap"
    if "cross_performance" in pipeline:
        return "cross_performance"
    if "composing" in pipeline:
        return "composing"
    if "random" in pipeline:
        return "random"
    raise ValueError(pipeline)


def read_experiment_results(experiment: str):
    pipeline_regex = re.compile(r"^(.*)_\d{4}-\d{2}-\d{2}-\d{2}:\d{2}$")

    rows = []
    for dataset, loss, loss_maximize in zip(datasets, losses, losses_maximize):
        experiment_date = find_latest_experiment_date(experiment, dataset)
        pipeline_groups = [
            dir
            for dir in os.listdir(f"../../data/results/{experiment}/{dataset}")
            if os.path.isdir(f"../../data/results/{experiment}/{dataset}/{dir}")
            if experiment_date in dir
        ]
        pipeline_groups_clean = [
            pipeline_regex.match(pipeline).group(1).replace("_uniform", "")
            for pipeline in pipeline_groups
        ]
        for pipeline_group, pipeline_group_clean in zip(
                pipeline_groups, pipeline_groups_clean
        ):
            pipelines = [
                dir
                for dir in os.listdir(
                    f"../../data/results/{experiment}/{dataset}/{pipeline_group}"
                )
                if os.path.isdir(
                    f"../../data/results/{experiment}/{dataset}/{pipeline_group}/{dir}"
                )
            ]
            for pipeline in pipelines:
                cluster_amounts = os.listdir(
                    f"../../data/results/{experiment}/{dataset}/{pipeline_group}/{pipeline}"
                )
                for cluster_amount in cluster_amounts:
                    with open(
                            f"../../data/results/{experiment}/{dataset}/{pipeline_group}/{pipeline}/{cluster_amount}/params.json"
                    ) as f:
                        params = json.load(f)
                        num_cluster = params[
                            "modelclusterer__cluster_mixin__n_clusters"
                        ]
                    with open(
                            f"../../data/results/{experiment}/{dataset}/{pipeline_group}/{pipeline}/{cluster_amount}/scores.json"
                    ) as f:
                        try:
                            scores = json.load(f)
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "loss": loss,
                                    "pipeline": pipeline,
                                    "pipeline_group": clean_pipeline_group(pipeline),
                                    **scores,
                                    "num_clusters": int(num_cluster),
                                }
                            )
                        except json.JSONDecodeError:
                            print(
                                f"Could not load scores for {dataset}/{pipeline_group}/{pipeline}/{cluster_amount}"
                            )
                            print(f)
                rows.append(
                    {
                        "dataset": dataset,
                        "pipeline": pipeline,
                        "pipeline_group": clean_pipeline_group(pipeline),
                        **scores,
                    }
                )
    return pd.DataFrame(rows), pipeline_groups, pipeline_groups_clean


def find_latest_experiment_date(experiment: str, dataset: str) -> str:
    experiment_path = f"../../data/results/{experiment}/{dataset}"
    if not os.path.exists(experiment_path):
        return None
    dates = [
        dir.split("_")[-1]
        for dir in os.listdir(experiment_path)
        if os.path.isdir(os.path.join(experiment_path, dir))
    ]
    if not dates:
        return None
    return max(dates, key=lambda d: d)