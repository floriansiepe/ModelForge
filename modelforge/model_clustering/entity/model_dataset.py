import random
import zlib
from functools import lru_cache
from typing import Iterable, List, Union

from .model_entity import ModelEntity


class ModelDataSet:
    _model_entities: dict[Union[int, str], ModelEntity]

    def __init__(self, model_entities: dict[Union[int, str], ModelEntity] = None):
        if model_entities is None:
            model_entities = {}
        self._model_entities = model_entities

    def model_entities(self) -> List[ModelEntity]:
        return [self.model_entity_by_id(key) for key in self.model_entity_ids()]

    def model_entity_ids(self) -> List[Union[int, str]]:
        return sorted(self._model_entities.keys(), key=str)

    def model_entity_by_id(self, entity_id: Union[int, str]):
        return self._model_entities.get(entity_id)

    def add_model_entity(self, model_entity: ModelEntity):
        if model_entity.id in self._model_entities:
            raise ValueError(f"Model entity {model_entity.id} already exists")
        self._model_entities[model_entity.id] = model_entity

    def sample(self, n: int, random_state: int = 42) -> "ModelDataSet":
        if n > self.size:
            raise ValueError(f"Cannot sample {n} from dataset of size {self.size}")
        random.seed(random_state)
        model_entities = random.sample(self.model_entities(), n)
        return ModelDataSet.from_iterable(model_entities)

    def train_test_split(
        self, train_size: float = 0.8, random_state: int = 42
    ) -> tuple["ModelDataSet", "ModelDataSet"]:
        if train_size < 0 or train_size > 1:
            raise ValueError("train_size must be between 0 and 1")
        random.seed(random_state)
        n_train = int(self.size * train_size)
        train_entities = random.sample(self.model_entities(), n_train)
        test_entities = [
            entity for entity in self.model_entities() if entity not in train_entities
        ]
        return ModelDataSet.from_iterable(train_entities), ModelDataSet.from_iterable(
            test_entities
        )

    @property
    @lru_cache(maxsize=1)
    def feature_list(self):
        if len(self._model_entities) == 0:
            raise ValueError("No entities in dataset")
        return self.model_entities()[0].feature_list

    @property
    @lru_cache(maxsize=1)
    def target(self):
        if len(self._model_entities) == 0:
            raise ValueError("No entities in dataset")
        return self.model_entities()[0].train_y.name

    @property
    def size(self):
        return len(self._model_entities)

    def __hash__(self):
        return zlib.adler32(
            str(sum([entity.__hash__() for entity in self.model_entities()])).encode()
        )

    @property
    def shape(self):
        return self.size, len(self.feature_list)

    @staticmethod
    def from_iterable(entities: Iterable[ModelEntity]):
        model_entities = {entity.id: entity for entity in entities}
        return ModelDataSet(model_entities)
