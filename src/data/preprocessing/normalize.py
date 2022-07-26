from typing import List

import numpy as np
import pandas as pd

from src.data.preprocessing.base import (
    FeatureTransformation,
    ColumnTransformation,
)


class ClipNormalize(FeatureTransformation):
    def __init__(self, max_value: int):
        self.max_value = max_value

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Clip features to max value: {self.max_value}")
        return df.clip(upper=self.max_value)


class MinMaxNormalize(FeatureTransformation):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"MinMax normalize features")
        df = (df - df.min()) / (df.max() - df.min())
        df[df.isna()] = 0
        return df


class QueryMinMaxNormalize(ColumnTransformation):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"MinMax normalize features per query")
        return df.groupby("query").apply(self.scale)

    def scale(self, df: pd.DataFrame):
        query = df["query"].values[0]
        df = (df - df.min()) / (df.max() - df.min())
        df[df.isna()] = 0
        df["query"] = query
        return df

    def get_columns(self, df) -> List[str]:
        return [c for c in df.columns if c not in ["y"]]


class Log1pNormalize(FeatureTransformation):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Log1p normalize features")
        return np.sign(df) * np.log(np.abs(df) + 1)


class ZScoreNormalize(FeatureTransformation):
    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Z-Score normalize features")
        # Epsilon in case of zero variance features
        return (df - df.mean()) / (df.std() + self.epsilon)
