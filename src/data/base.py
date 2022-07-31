from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd
import torch

from src.data.preprocessing import Pipeline


class DatasetLoader(ABC):
    def __init__(self, name: str, fold: int, n_features: int, n_results: int, pipeline: Pipeline):
        self.name = name
        self.fold = fold
        self.n_features = n_features
        self.n_results = n_results
        self.pipeline = pipeline

        assert fold in self.folds

    @property
    def cache_directory(self):
        path = Path.home() / ".ltr_datasets" / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def preprocess_directory(self):
        path = Path.home() / ".ltr_datasets" / "processed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def dataset_directory(self):
        path = Path.home() / ".ltr_datasets" / "dataset"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def download_directory(self):
        path = Path.home() / ".ltr_datasets" / "download"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def feature_columns(self):
        return list(map(str, range(self.n_features)))

    def load(self, split: str) -> pd.DataFrame:
        assert split in self.splits, f"Split must one of {self.splits}"
        cache_path = self.preprocess_directory / f"{self.name}-{self.fold}-{split}.pt"

        if not cache_path.exists():
            path = self.cache_directory / f"{self.name}-{self.fold}-{split}.parquet"

            if not path.exists():
                df = self._parse(split)
                df.to_parquet(path)

            df = pd.read_parquet(path)
            dataset = self.pipeline(df)

            torch.save(dataset, cache_path)

        return torch.load(cache_path)




    @property
    @abstractmethod
    def folds(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def splits(self) -> List[str]:
        pass

    @abstractmethod
    def _parse(self, split: str) -> pd.DataFrame:
        pass
