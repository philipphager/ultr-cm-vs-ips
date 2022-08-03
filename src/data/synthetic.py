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
        pipeline: Pipeline,
    ):
        super().__init__(name, fold, n_features, n_results, pipeline)

    def load(self, split: str):
        n_features = 1000

        n_queries = int(self.n_features / self.n_results)
        relevance = torch.rand(n_features)
        q = torch.arange(n_queries)
        n = torch.full((n_queries,), self.n_results)

        seed = abs(hash(split)) % (10**8)
        generator = torch.Generator()
        generator.manual_seed(seed)

        if split == "train":
            # Create random train queries containing each document
            shape = (n_queries, self.n_results)
            x = torch.randperm(n_features, generator=generator).reshape(shape)
        else:
            # Create random val / test queries containing a sample of documents
            shape = (int(n_queries), self.n_results)
            x = torch.randint(n_features, shape, generator=generator)

        y = relevance[x]
        x = F.one_hot(x)

        return RatingDataset(q, n, x, y)

    def _parse(self, split: str):
        pass

    @property
    def folds(self) -> List[int]:
        return [1]

    @property
    def splits(self) -> List[str]:
        return ["train", "val", "test"]
