import zlib
from logging import Logger
from typing import Callable, List

import numpy as np
import pandas as pd
from distributed import Client, LocalCluster
from joblib import Memory
from sklearn.utils.validation import check_is_fitted

from modelforge.model_clustering.distances.cross_performance import compute_loss
from modelforge.model_clustering.entity.loss import LossFunction
from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.transformer.featurizer.featurizer import Featurizer
from modelforge.model_clustering.transformer.sampler.set.set_sampler import (
    SetSampler,
)
from modelforge.shared.logger import logger_factory


def prepare_features(
    client: Client,
    logger: Logger,
    loss_function: Callable,
    model_dataset: ModelDataSet,
    row_ids: List[str],
    column_ids: List[str],
    use_train: bool,
    # The following arguments are used to hash the function and dataset
    _loss_function_hash: int,
    _model_dataset_hash: int,
    _row_ids_hash: int,
    _column_ids_hash: int,
):
    def prepare_device_feature(
        model_id: str, loss_function: callable, index: int, ids: List[str | int]
    ) -> np.ndarray:
        # Each series is the embedding for a single device
        logger.debug(f"{index}/{len(ids)} Preparing features for {model_id}")

        embedding = np.zeros(len(ids))
        for i, test_id in enumerate(ids):
            test_entity = model_dataset.model_entity_by_id(test_id)
            model_entity = model_dataset.model_entity_by_id(model_id)
            embeddings_component = compute_loss(
                model_entity, test_entity, loss_function, train=use_train
            )
            embedding[i] = embeddings_component

        return embedding

    results = []
    i = 1

    for model_id in row_ids:
        embedding = client.submit(
            prepare_device_feature, model_id, loss_function, i, column_ids
        )
        results.append(embedding)
        i += 1
    logger.info(f"Submitted {i} tasks to the compute pair-wise loss")

    gathered_results = client.gather(results)
    # Create a dataframe with each row being the embedding for a single device
    embedding_df = pd.DataFrame(gathered_results, index=row_ids, columns=column_ids)
    return embedding_df


class PredictionLossSetFeaturizer(Featurizer):
    """
    Featurize set dataset by prediction loss on each other's training or testing set. For each set $m_B$ in the dataset,
    there will be a component in the feature vector of set $m_A$ such that it reflects the loss of $m_A$ on the
    test set of $m_B$. This is done for all models in the dataset. Hence, the feature vector of set $m_A$ will be
    of length $n$ where $n$ is the number of models in the dataset.

    @param loss_function: Loss function to compute the loss between two models
    """

    def __init__(
        self,
        loss_function: LossFunction = None,
        model_sampler: SetSampler = None,
        memory: Memory = None,
        client: Client = None,
        skip_cache: bool = False,
        use_train: bool = True,
    ):
        super().__init__()
        self.loss_function = loss_function
        self.logger = logger_factory(__name__)
        self.memory = memory
        if memory is None:
            self.memory = Memory(location="./.cache/", verbose=0)
        if client is None:
            client = LocalCluster(n_workers=1, threads_per_worker=1).get_client()
        self.client = client
        self.model_sampler = model_sampler
        self.column_ids_ = None
        self.row_ids_ = None
        self.skip_cache = skip_cache
        self.use_train = use_train

    def fit(self, model_dataset: ModelDataSet, _y=None):
        self.row_ids_ = model_dataset.model_entity_ids()
        if self.model_sampler is not None:
            self.column_ids_ = sorted(
                [
                    model_entity.id
                    for model_entity in self.model_sampler.get_sample_candidates(
                        model_dataset
                    )
                ],
                key=str,
            )
        else:
            self.column_ids_ = model_dataset.model_entity_ids()

        return self

    def featurize(self, model_dataset: ModelDataSet) -> pd.DataFrame:
        """
        Transform the set dataset into a feature matrix

        @param model_dataset: The set dataset to be featurized
        @return: A feature matrix
        """
        check_is_fitted(self)
        loss_function = (
            self.loss_function
            if self.loss_function is not None
            else model_dataset.model_entities()[0].loss
        )
        if self.skip_cache:
            return prepare_features(
                self.client,
                self.logger,
                loss_function,
                model_dataset,
                self.row_ids_,
                self.column_ids_,
                self.use_train,
                0,
                0,
                0,
                0,
            )

        # Joblib memory cache does not support hashing of functions or non numpy object, so we need to pass the hash
        # of the function and dataset as arguments, which are not used in the function. The original arguments of the
        # loss function and set dataset are ignored by joblib.memory.
        # Moreover, memory only supports pure functions and not methods, so we delegate the method to a function.
        loss_function_hash = zlib.adler32(loss_function.__name__.encode())
        model_dataset_hash = model_dataset.__hash__()
        row_ids = zlib.adler32(str(self.row_ids_).encode())
        column_ids = zlib.adler32(str(self.column_ids_).encode())
        row_ids_hash = zlib.adler32(str(row_ids).encode())
        column_ids_hash = zlib.adler32(str(column_ids).encode())

        return self.memory.cache(
            prepare_features,
            ignore=[
                "client",
                "logger",
                "loss_function",
                "model_dataset",
                "row_ids",
                "column_ids",
            ],
        )(
            self.client,
            self.logger,
            loss_function,
            model_dataset,
            self.row_ids_,
            self.column_ids_,
            self.use_train,
            loss_function_hash,
            model_dataset_hash,
            row_ids_hash,
            column_ids_hash,
        )

    def __repr__(self, n_char_max=700):
        loss_function = ""
        if self.loss_function is not None:
            loss_function = self.loss_function.__name__
        return f"PairwiseLossFeaturizer(loss_function={loss_function}, model_sampler={self.model_sampler.__str__()})"

    def __str__(self):
        return self.__repr__()
