import os
from collections import defaultdict
from pathlib import Path
from typing import List

import altair as alt
import pandas as pd
import torch


def _parse_hydra_config(path: Path):
    import yaml

    # Load yaml file ignoring custom hydra attributes
    file = open(path, "r")
    raw_yaml = yaml.full_load(file)
    return pd.json_normalize(raw_yaml, sep="_")


def _subset(frame, columns: List[str], metric_columns: List[str]):
    return (
        frame[columns + metric_columns]
        .dropna(axis=0, how="all", subset=metric_columns)
        .copy()
    )


def _cross_join(df1, df2):
    df1["key"] = 1
    df2["key"] = 1
    return df1.merge(df2, on="key").drop(columns=["key"])


def _rename(frame, stage):
    return frame.rename(
        columns={
            "model_name": "model",
            "data_name": "dataset",
            "simulation_n_sessions": "n_sessions",
            f"{stage}_ndcg": "nDCG",
            f"{stage}_ndcg@5": "nDCG@5",
            f"{stage}_ndcg@10": "nDCG@10",
            f"{stage}_arp": "ARP",
        }
    )


def load_experiment(experiment: str, directory: Path = Path("results")):
    path = directory / experiment
    frames = defaultdict(lambda: [])

    if not path.exists():
        return None, None

    for directory in path.iterdir():
        baseline_path = directory / "baseline.parquet"
        val_path = directory / "val.parquet"
        test_path = directory / "test.parquet"

        if directory.is_dir() and baseline_path.exists() and val_path.exists() and test_path.exists():
            param_df = _parse_hydra_config(directory / "config.yaml")
            baseline_df = pd.read_parquet(baseline_path)
            val_df = pd.read_parquet(val_path)
            test_df = pd.read_parquet(test_path)
            test_df["dir"] = str(directory)

            baseline_df = _cross_join(baseline_df, param_df)
            val_df = _cross_join(val_df, param_df)
            test_df = _cross_join(test_df, param_df)

            frames["baseline"].append(baseline_df)
            frames["val"].append(val_df)
            frames["test"].append(test_df)

    baseline_df = (
        _rename(pd.concat(frames["baseline"]), "test") if len(frames["baseline"]) > 0 else None
    )
    val_df = (
        _rename(pd.concat(frames["val"]), "val") if len(frames["val"]) > 0 else None
    )
    test_df = (
        _rename(pd.concat(frames["test"]), "test") if len(frames["test"]) > 0 else None
    )

    return baseline_df, val_df, test_df
