from typing import List

import torch
import torch.nn.functional as F

from src.data.base import DatasetLoader
from src.data.preprocessing import Pipeline
from src.data.preprocessing.convert import RatingDataset


class Synthetic(DatasetLoader):
    def __init__(
        self,
        name: str,
        fold: int,
        n_features: int,
        n_results: int,
        perc_feature_collision: float,
        pipeline: Pipeline,
    ):
        super().__init__(name, fold, n_features, n_results, pipeline)
        self.perc_feature_collision = perc_feature_collision

    def load(self, split: str):
        generator = torch.Generator()
        generator.manual_seed(0)

        n_queries = int(self.n_features * 2 / self.n_results)
        relevance = torch.randint(5, (self.n_features,), generator=generator)
        q = torch.arange(n_queries)
        n = torch.full((n_queries,), self.n_results)

        seed = abs(hash(split)) % (10**8)
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Create random queries with two occurrences of each document.
        shape = (n_queries, self.n_results)
        x = torch.randperm(self.n_features * 2, generator=generator).reshape(shape)
        x = x % self.n_features

        max_features = int(self.n_features * (1 - self.perc_feature_collision))

        y = relevance[x]
        x = F.one_hot(x % max_features, num_classes=self.n_features).float()

        return RatingDataset(q, n, x, y)

    def _parse(self, split: str):
        pass

    @property
    def folds(self) -> List[int]:
        return [1]

    @property
    def splits(self) -> List[str]:
        return ["train", "val", "test"]
