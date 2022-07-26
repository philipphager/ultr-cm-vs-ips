import pandas as pd

from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only


class LocalLogger(LightningLoggerBase):
    def __init__(self):
        self.results = {"val": [], "test": []}

    @property
    def name(self):
        return "logger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if "val_loss" in metrics:
            self.results["val"].append(metrics)
        elif "test_loss" in metrics:
            self.results["test"].append(metrics)

    @rank_zero_only
    def save(self):
        for stage, rows in self.results.items():
            df = pd.DataFrame(rows)
            df.to_parquet(f"{stage}.parquet")

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass

    def should_log(self, metrics):
        return any([f"{s}_loss" in metrics for s in self.steps])
