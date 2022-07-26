from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing.util import get_feature_columns
from src.util.tensor import print_memory_footprint


class RatingDataset(Dataset):
    def __init__(
        self,
        q: torch.Tensor,
        n: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        self.q = q
        self.n = n
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i: int):
        return self.q[i], self.n[i], self.x[i], self.y[i]


class ToTorch:
    def __call__(self, df: pd.DataFrame) -> RatingDataset:
        print("Converting DataFrame to torch tensors")
        feature_columns = get_feature_columns(df)

        query_df = df.groupby(["query"]).agg(n=("y", "count")).reset_index()
        feature_df = self._group_by_query(df, feature_columns, "x")
        target_df = self._group_by_query(df, "y", "y")

        query_df = query_df.merge(feature_df, on=["query"])
        query_df = query_df.merge(target_df, on=["query"])

        n_queries = len(query_df)
        n_results = query_df["n"].max()
        n_features = len(feature_columns)

        print(f"Creating tensors with max number of documents: {n_results}")
        x = torch.zeros((n_queries, n_results, n_features), dtype=torch.float)
        y = torch.zeros((n_queries, n_results), dtype=torch.long)
        q = torch.zeros((n_queries,), dtype=torch.long)
        n = torch.zeros((n_queries,), dtype=torch.long)

        for i, row in query_df.iterrows():
            q[i] = row["query"]
            n[i] = row["n"]
            x[i, : row["n"]] = torch.from_numpy(row["x"])
            y[i, : row["n"]] = torch.from_numpy(row["y"])

        print("\n##### MEMORY #####")
        print_memory_footprint(n, "n")
        print_memory_footprint(q, "query")
        print_memory_footprint(x, "x")
        print_memory_footprint(y, "y")

        return RatingDataset(q, n, x, y)

    @staticmethod
    def _group_by_query(
        df: pd.DataFrame, columns: Union[str, List[str]], name: str
    ) -> pd.DataFrame:
        return (
            df.groupby(["query"])
            .apply(lambda x: x[columns].values)
            .rename(name)
            .reset_index()
        )
