from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in ["y", "query"]]


def random_split(
    dataset: Dataset, frac: Optional[float] = None, n: Optional[int] = None
):
    assert bool(frac) != bool(n), "Use either frac or n"

    train_size = int(len(dataset) * frac) if frac is not None else n
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])
