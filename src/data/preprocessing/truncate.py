import pandas as pd

from src.data.preprocessing.base import Transformation


class UniformTruncate(Transformation):
    def __init__(self, max_length: int, random_state: int):
        self.max_length = max_length
        self.random_state = random_state

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Uniformly sampling queries to max {self.max_length} documents")
        return df.groupby(["query"], group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), self.max_length), random_state=self.random_state
            )
        )


class StratifiedTruncate(Transformation):
    def __init__(self, max_length: int, random_state: int):
        self.max_length = max_length
        self.random_state = random_state

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Stratified sampling queries to max {self.max_length} documents")
        return df.groupby(["query"], group_keys=False).apply(self.stratified_sample)

    def stratified_sample(self, df):
        n_results = min(self.max_length, len(df))
        return (
            df.groupby("y")
            .sample(frac=n_results / len(df), random_state=self.random_state)
            .tail(n_results)
        )


class CutoffTruncate(Transformation):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Cutting off queries after max {self.max_length} documents")

        return df.groupby(["query"], group_keys=False).apply(
            lambda x: x.head(min(len(x), self.max_length))
        )
