import logging
from typing import List, Union, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn

from src.metric import get_metrics
from src.model.base import create_sequential
from src.simulation.bias import get_position_bias


class NeuralPBM(LightningModule):
    def __init__(
        self,
        name: str,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_features: int,
        n_results: int,
        position_bias: Optional[float],
        layers: Union[List[int], str],
        dropouts: Union[List[float], str],
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
        self.relevance = nn.Sequential(
            create_sequential(n_features, layers, dropouts, activation), nn.Sigmoid()
        )

        if position_bias is not None:
            position_bias = get_position_bias(n_results, position_bias)
            logging.debug(f"Using pre-defined position bias: {position_bias[:5]}")
            self.examination = nn.Embedding.from_pretrained(position_bias.unsqueeze(-1))
        else:
            logging.debug("No position bias specified, inferring bias...")
            self.examination = nn.Sequential(nn.Embedding(n_results, 1), nn.Sigmoid())

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adagrad":
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch, n_items, _ = x.shape

        ranks = torch.arange(n_items, device=self.device).repeat(n_batch, 1)
        examination = self.examination(ranks)
        relevance = self.relevance(x)
        y_predict = examination * relevance

        return y_predict.squeeze(), relevance.squeeze()

    def training_step(self, batch, idx):
        q, n, x, y, y_click = batch  # ClickDataset

        y_predict, _ = self.forward(x)
        loss = self.loss(y_predict, y_click, n)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, idx):
        q, n, x, y, y_click = batch  # ClickDataset

        y_predict_click, y_predict = self.forward(x)
        loss = self.loss(y_predict_click, y_click, n)
        metrics = get_metrics(y_predict, y, n, "val_")

        self.log("val_loss", loss)
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, idx):
        q, n, x, y = batch  # RatingDataset

        y_predict_click, y_predict = self.forward(x)
        metrics = get_metrics(y_predict, y, n, "test_")

        self.log_dict(metrics)
        return metrics

    def get_position_bias(self):
        ranks = torch.arange(self.n_results, device=self.device)
        return self.examination(ranks).squeeze()
