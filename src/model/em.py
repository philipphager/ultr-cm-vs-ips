from typing import Optional, List, Union

import torch
from pytorch_lightning import LightningModule
from torch import nn

from src.metric import get_metrics
from src.model.base import create_sequential
from src.simulation.bias import get_position_bias


class RegressionEM(LightningModule):
    def __init__(
        self,
        name: str,
        n_features: int,
        n_results: int,
        position_bias: Optional[float],
        layers: Union[List[int], str],
        dropouts: Union[List[float], str],
        activation: nn.Module,
        learning_rate: float,
        em_step_size: float,
        loss: nn.Module,
        optimizer: str,
    ):

        super().__init__()
        self.name = name
        self.n_results = n_results
        self.loss = loss
        self.learning_rate = learning_rate
        self.em_step_size = em_step_size
        self.optimizer = optimizer
        self.relevance = create_sequential(n_features, layers, dropouts, activation)

        if position_bias is not None:
            self.examination = get_position_bias(n_results, position_bias).to(
                self.device
            )
            self.perform_maximization_step = False
        else:
            print("No position bias specified, inferring bias through EM...")
            self.examination = torch.full((n_results,), 0.5).to(self.device)
            self.perform_maximization_step = True

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adagrad":
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(
                f"Expected optimizer: adam, adagrad but found: {self.optimizer}"
            )

    def training_step(self, batch):
        q, n, x, y, y_click = batch  # ClickDataset
        n_batch = x.size(0)

        examination = self.examination.repeat(n_batch, 1)
        relevance = torch.sigmoid(self.relevance(x)).squeeze()

        p_e1_ro_c0 = examination * (1 - relevance) / (1 - examination * relevance)
        p_e0_r1_c0 = (1 - examination) * relevance / (1 - examination * relevance)

        # Marginalize document relevance:
        # P(r = 1) = P(c = 1) * P(e = 1, r = 1 | c = 1)
        # + P(c = 0) * P(e = 0, r = 1 | c = 0)
        p_r1 = y_click + (1 - y_click) * p_e0_r1_c0

        # Instead of directly regressing on p_r1, the original paper
        # uses bernoulli samples and trains a LGBM classifier for document relevance.
        # Thus, we fit our relevance network to samples here.
        relevance_samples = torch.bernoulli(p_r1)
        loss = self.loss(relevance, relevance_samples.detach(), n)

        if self.perform_maximization_step:
            with torch.no_grad():
                # Marginalized probability of rank examination
                p_e1 = (y_click + (1 - y_click) * p_e1_ro_c0).mean(0)
                self.examination = (
                    1 - self.em_step_size
                ) * self.examination + self.em_step_size * p_e1

        return loss

    def validation_step(self, batch, idx):
        q, n, x, y, y_click = batch  # ClickDataset

        y_predict = torch.sigmoid(self.relevance(x)).squeeze()
        metrics = get_metrics(y_predict, y, n, "val_")
        self.log_dict(metrics)

        # We fit this model until convergence on a fixed number of epochs and
        # do not employ early stopping based on val loss.
        loss = 0

        return loss, metrics

    def test_step(self, batch, idx):
        q, n, x, y = batch  # RatingDataset

        y_predict = torch.sigmoid(self.relevance(x)).squeeze()
        metrics = get_metrics(y_predict, y, n, "test_")

        self.log_dict(metrics)
        return metrics

    def get_position_bias(self) -> torch.Tensor:
        return self.examination.detach()
