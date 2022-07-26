from typing import List

import torch
from pytorch_lightning import LightningModule
from torch import nn

from src.metric import get_metrics
from src.model.base import create_sequential
from src.simulation.bias import get_position_bias


class IPS(LightningModule):
    def __init__(
        self,
        name: str,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_features: int,
        n_results: int,
        position_bias: float,
        layers: List[int],
        dropouts: List[float],
        activation: nn.Module,
    ):
        super().__init__()
        self.name = name
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.layers = layers
        self.n_features = n_features
        self.n_results = n_results
        self.relevance = create_sequential(n_features, layers, dropouts, activation)
        self.register_buffer(
            "position_bias", get_position_bias(n_results, position_bias)
        )

        # self.save_hyperparameters()

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adagrad":
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relevance(x).squeeze()

    def predict_step(self, batch, idx, **kwargs):
        x, y, y_click, n = batch
        return self.forward(x)

    def training_step(self, batch, idx):
        x, y, y_click, n = batch
        _, n_results = y.shape

        y_predict = self.forward(x)
        loss = self.loss(y_predict, y_click, n, self.position_bias[:n_results])

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, idx):
        x, y, y_click, n = batch
        _, n_results = y.shape

        y_predict = self.forward(x)
        loss = self.loss(y_predict, y_click, n, self.position_bias[:n_results])
        metrics = get_metrics(torch.sigmoid(y_predict), y, n, "val_")

        self.log("val_loss", loss)
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, idx):
        x, y, y_click, n = batch
        _, n_results = y.shape

        y_predict = self.forward(x)
        loss = self.loss(y_predict, y_click, n, self.position_bias[:n_results])
        metrics = get_metrics(torch.sigmoid(y_predict), y, n, "test_")

        self.log("test_loss", loss)
        self.log_dict(metrics)
        return metrics
