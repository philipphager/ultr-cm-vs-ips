import pandas as pd

from src.data.preprocessing.base import Transformation


class FilterShortQueries(Transformation):
    def __init__(self, min_length: int):
        self.min_length = min_length

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        query_df = df.groupby("query").agg(n=("y", "count")).reset_index()
        filter_df = query_df[query_df["n"] < self.min_length]

        if len(filter_df) > 0:
            print(
                f"Discarding {len(filter_df)} / {len(query_df)} queries with less than "
                f"{self.min_length} documents"
            )

        return df[~df["query"].isin(filter_df["query"])]


class FilterNonRelevant(Transformation):
    def __init__(self, min_relevance: int):
        self.min_relevance = min_relevance

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        query_df = df.groupby("query").agg(y_max=("y", "max")).reset_index()
        filter_df = query_df[query_df["y_max"] < self.min_relevance]

        if len(filter_df) > 0:
            print(
                f"Discarding {len(filter_df)} / {len(query_df)} queries without a "
                f"single document of relevance {self.min_relevance}"
            )

        return df[~df["query"].isin(filter_df["query"])]
