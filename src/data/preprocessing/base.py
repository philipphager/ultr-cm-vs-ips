from abc import ABC, abstractmethod
from typing import List, Union

import pandas as pd
from omegaconf import DictConfig

from src.data.preprocessing.convert import ToTorch, RatingDataset
from src.data.preprocessing.util import get_feature_columns
from src.util.functional import apply


class Transformation(ABC):
    @abstractmethod
    def __call__(self, df) -> pd.DataFrame:
        pass


class ColumnTransformation(Transformation, ABC):
    def __call__(self, df):
        columns = self.get_columns(df)
        df.loc[:, columns] = self.transform(df[columns])
        return df

    @abstractmethod
    def transform(self, df) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_columns(self, df) -> List[str]:
        pass


class FeatureTransformation(ColumnTransformation, ABC):
    def get_columns(self, df) -> List[str]:
        return get_feature_columns(df)


class Pipeline:
    def __init__(
        self,
        normalize: List[ColumnTransformation] = [],
        truncate: List[Transformation] = [],
        filter: List[Transformation] = [],
    ):
        self.normalize = self.from_omega_conf(normalize)
        self.truncate = self.from_omega_conf(truncate)
        self.filter = self.from_omega_conf(filter)
        self.convert = ToTorch()

    def __call__(self, df: pd.DataFrame) -> RatingDataset:
        print("\n##### PREPROCESSING #####")
        print("\n##### STATS BEFORE #####")
        self.print_stats(df)

        print("\n##### STEPS #####")
        df = apply(self.normalize, df)
        df = apply(self.truncate, df)
        df = apply(self.filter, df)
        dataset = self.convert(df)

        print("\n##### STATS AFTER #####")
        self.print_stats(df)

        return dataset

    def print_stats(self, df: pd.DataFrame):
        self._print_dataset_size(df)
        self._print_relevance_distribution(df)

    @staticmethod
    def _print_dataset_size(df: pd.DataFrame):
        query_df = df.groupby("query").agg(n=("y", "count"))
        print(f"Queries: {len(query_df)}")
        print(f"Documents: {query_df.n.sum()}")
        print(
            "Documents per Query:\n",
            query_df.reset_index()
            .n.describe(percentiles=[0.5, 0.9, 0.99])
            .reset_index(),
        )

    @staticmethod
    def _print_relevance_distribution(df: pd.DataFrame):
        stats_df = df.groupby(["y"]).agg(perc_docs=("y", "count")).reset_index()
        stats_df.perc_docs /= stats_df.perc_docs.sum()
        print("\n% of Documents with Relevance:\n", stats_df)

    @staticmethod
    def from_omega_conf(param: Union[List, DictConfig]) -> List[Transformation]:
        if isinstance(param, DictConfig):
            return list(param.values())
        return param
