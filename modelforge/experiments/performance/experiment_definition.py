import json
import logging
import os
import shutil
import time

import click
import numpy as np
from distributed import LocalCluster

from modelforge.datasets.datasets import (
    anomaly_dataset,
    heating_dataset,
    immoscout_house_price_dataset,
    weather_dataset_probabilistic_regression,
)
from modelforge.model_clustering.distances.cross_performance import (
    CrossPerformanceDistance,
)
from modelforge.model_clustering.transformer.featurizer.prediction_loss_set_featurizer import (
    PredictionLossSetFeaturizer,
)
from modelforge.model_clustering.transformer.sampler.set.criterion_selector.target_selector import (
    TargetSelector,
)
from modelforge.model_clustering.transformer.sampler.set.measure.entropy import Entropy
from modelforge.model_clustering.transformer.sampler.set.statistical_set_sampler import (
    StatisticalSetSampler,
)
from modelforge.shared.losses import crps_unpacking


@click.command()
@click.option(
    "--num-runs",
    default=3,
    help="Number of times to run each step for timing measurement",
)
def main(num_runs):
    shutil.rmtree("data/results/performance", ignore_errors=True)
    os.makedirs(f"data/results/performance", exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler
    log_file_path = "logs/performance_experiment.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Create formatter and add to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Running each step {num_runs} times for robust timing measurements")

    # Dictionary to store all timing results
    timing_results = {}

    heating = heating_dataset()
    house_price = immoscout_house_price_dataset()
    weather_probabilistic = weather_dataset_probabilistic_regression()
    anomaly = anomaly_dataset("anomaly")
    datasets = [
        heating,
        house_price,
        weather_probabilistic,
        anomaly,
    ]
    dataset_names = [
        "heating",
        "house_price",
        "weather_probabilistic",
        "anomaly",
    ]
    client = LocalCluster(n_workers=1, threads_per_worker=1).get_client()

    for names, dataset in zip(dataset_names, datasets):
        logger.info(f"Starting dataset {names}")

        loss = dataset.model_entities()[0].loss
        if names == "weather_probabilistic":
            loss = crps_unpacking
        sampler = StatisticalSetSampler(10, TargetSelector(), Entropy(), "max")
        use_train = True
        # Quick fix as many training sets have no anomalies
        if names == "anomaly":  # Anomaly dataset
            use_train = False

        # Store timing measurements for multiple runs
        featurize_times = []
        transform_times = []

        for run in range(num_runs):
            logger.info(f"Starting run {run + 1}/{num_runs} for dataset {names}")

            # Measure featurize step
            featurizer = PredictionLossSetFeaturizer(
                model_sampler=sampler,
                loss_function=loss,
                use_train=use_train,
                skip_cache=True,
                client=client,
            )
            logger.info(f"Starting featurization for {names} (run {run + 1})")
            start_time = time.time()
            featurizer.fit_transform(dataset)
            featurize_time = time.time() - start_time
            featurize_times.append(featurize_time)
            logger.info(
                f"Featurization for {names} (run {run + 1}) took {featurize_time:.2f} seconds"
            )

            # Measure transform step
            cp = CrossPerformanceDistance(
                skip_cache=True, client=client, loss_function=loss
            )
            logger.info(f"Starting transformation for {names} (run {run + 1})")
            start_time = time.time()
            cp.transform(dataset)
            transform_time = time.time() - start_time
            transform_times.append(transform_time)
            logger.info(
                f"Transformation for {names} (run {run + 1}) took {transform_time:.2f} seconds"
            )

        # Calculate statistics
        featurize_mean = np.mean(featurize_times)
        featurize_std = np.std(featurize_times, ddof=1)  # Sample standard deviation
        transform_mean = np.mean(transform_times)
        transform_std = np.std(transform_times, ddof=1)

        total_times = [f + t for f, t in zip(featurize_times, transform_times)]
        total_mean = np.mean(total_times)
        total_std = np.std(total_times, ddof=1)

        logger.info(f"Dataset {names} summary:")
        logger.info(
            f"  Featurization: {featurize_mean:.2f} ± {featurize_std:.2f} seconds"
        )
        logger.info(
            f"  Transformation: {transform_mean:.2f} ± {transform_std:.2f} seconds"
        )
        logger.info(f"  Total: {total_mean:.2f} ± {total_std:.2f} seconds")

        # Create pandas-friendly timing record with statistics
        dataset_timing = {
            "dataset": names,
            "num_runs": num_runs,
            "featurize_mean": featurize_mean,
            "featurize_std": featurize_std,
            "featurize_min": np.min(featurize_times),
            "featurize_max": np.max(featurize_times),
            "transform_mean": transform_mean,
            "transform_std": transform_std,
            "transform_min": np.min(transform_times),
            "transform_max": np.max(transform_times),
            "total_mean": total_mean,
            "total_std": total_std,
            "total_min": np.min(total_times),
            "total_max": np.max(total_times),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "num_models": len(dataset.model_entities()),
            # Also store raw timing data for detailed analysis
            "featurize_times": featurize_times,
            "transform_times": transform_times,
            "total_times": total_times,
        }

        # Store timing results for this dataset
        timing_results[names] = dataset_timing

        logger.info(f"Finished dataset {names}")

    # Create pandas-friendly format (list of records)
    pandas_timing_results = []
    for result in timing_results.values():
        # Create a copy without the raw timing arrays for the main CSV
        result_clean = {k: v for k, v in result.items() if not k.endswith("_times")}
        pandas_timing_results.append(result_clean)

    import pandas as pd

    df = pd.DataFrame(pandas_timing_results)
    df.to_csv("data/results/performance/timing_results.csv", index=False)

    # Save detailed results including raw timing data as JSON
    with open("data/results/performance/detailed_timing_results.json", "w") as f:
        json.dump(timing_results, f, indent=2)

    logger.info("All timing results saved to data/results/performance/")


if __name__ == "__main__":
    main()
