from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class BinaryCrossEntropy(nn.Module):
    def forward(
        self,
        y_predict: torch.Tensor,
        y_true: torch.Tensor,
        n: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        clip: Optional[float] = 0.01,
        eps: float = 1e-10,
    ) -> torch.float:
        """
        Binary Cross-Entropy with IPS as in Bekker2019, Saito2020, Oosterhuis2022
        https://arxiv.org/pdf/2203.17118.pdf
        https://arxiv.org/pdf/1909.03601.pdf

        Args:
            y_predict: Tensor of size (n_batch, n_results) with predicted relevance
            y_true: Tensor of size (n_batch, n_results) with ground_truth scores
            position_bias: Tensor of size (n_results) with propensities per rank
            clip: Min propensity used to clip position_bias
            eps: Min value to avoid ln(0) = -inf

        Returns:
            Mean aggregated loss for the given batch
        """
        if position_bias is None:
            position_bias = torch.ones_like(y_true)

        if clip is not None:
            position_bias = position_bias.clip(min=clip)

        y_predict = torch.sigmoid(y_predict)
        position_bias = position_bias.type_as(y_predict)

        loss = -(
            (y_true / position_bias) * torch.log(y_predict.clip(min=eps))
            + (1 - (y_true / position_bias)) * torch.log((1 - y_predict).clip(min=eps))
        )

        loss = mask_padding(loss, n)
        return loss.sum(dim=1).mean()


class PairwiseHinge(nn.Module):
    def forward(
        self,
        y_predict: torch.Tensor,
        y_true: torch.Tensor,
        n: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        clip: Optional[float] = 0.01,
        margin: float = 1.0,
    ) -> torch.float:
        """

        Pairwise Hinge-Loss as used in SVMRank by Joachims2002.
        Optional debiasing by position using IPS scoring as in Joachims2017,Agarwal2019
        https://www.cs.cornell.edu/people/tj/publications/agarwal_etal_19b.pdf

        loss = sum_{y_true_i > y_true _j} max(0, 1 - (y_predict_i - y_predict_j))

        Args:
            y_predict: Tensor of size (n_batch, n_results) with predicted relevance
            y_true: Tensor of size (n_batch, n_results) with ground_truth scores
            position_bias: Tensor of size (n_results) with propensities per rank

        Returns:
            Mean aggregated loss for the given batch
        """
        n_batch = y_true.shape[0]

        y_true = mask_padding(y_true, n)
        y_predict = mask_padding(y_predict, n)

        y_predict_pairs = batch_pairs(y_predict)
        y_pairs = batch_pairs(y_true)

        y_predict_diff = y_predict_pairs[:, :, :, 0] - y_predict_pairs[:, :, :, 1]
        y_diff = y_pairs[:, :, :, 0] - y_pairs[:, :, :, 1]

        loss = margin - y_predict_diff
        loss[y_diff <= 0.0] = 0.0
        loss[loss < 0.0] = 0.0

        loss = loss.sum(-1)

        if position_bias is not None:
            if clip is not None:
                position_bias = position_bias.clip(min=clip)

            position_bias = position_bias.repeat(n_batch, 1)
            loss = loss / position_bias

        return loss.sum(dim=1).mean()


class ListNet(nn.Module):
    def forward(
        self,
        y_predict: torch.Tensor,
        y_true: torch.Tensor,
        n: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        clip: Optional[float] = 0.01,
        eps: float = 1e-10,
    ) -> torch.float:
        """
        ListNet Softmax loss by Cao2007:
        https://dl.acm.org/doi/10.1145/1273496.1273513

        Args:
            y_predict: Tensor of size (n_batch, n_results) with predicted relevance
            y_true: Tensor of size (n_batch, n_results) with ground_truth scores
            position_bias: Tensor of size (n_results) with propensities per rank
            eps: Min value to avoid ln(0) = -inf

        Returns:
            Mean aggregated loss for the given batch
        """

        if position_bias is not None:
            if clip is not None:
                position_bias = position_bias.clip(min=clip)

            n_batch = y_true.shape[0]
            y_true = y_true / position_bias.repeat(n_batch, 1)

        y_predict = mask_padding(y_predict, n, fill=-torch.inf)
        y_true = mask_padding(y_true, n, fill=-torch.inf)

        y_true = F.softmax(y_true.float(), dim=1)
        y_predict = F.softmax(y_predict, dim=1)
        y_predict = torch.log(y_predict + eps)

        loss = -(y_true * y_predict)
        return loss.sum(dim=1).mean()


def batch_pairs(x: torch.Tensor) -> torch.Tensor:
    """
    Creates i x j document pairs from batch or results.
    Adopted from pytorchltr

    Example:
        x = [
            [1, 2],
            [3, 4],
        ]

        [
            [[[1, 1], [1, 2]], [[2, 1], [2, 2]]],
            [[[3, 3], [3, 4]], [[4, 3], [4, 4]]]
        ]

    Args:
        x: Tensor of size (n_batch, n_results)

    Returns:
        Tensor of size (n_batch, n_results, n_results, 2) with all combinations
        of n_results.
    """

    if x.dim() == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))

    x_ij = torch.repeat_interleave(x, x.shape[1], dim=2)
    x_ji = torch.repeat_interleave(x.permute(0, 2, 1), x.shape[1], dim=1)

    return torch.stack([x_ij, x_ji], dim=3)


def mask_padding(x: torch.Tensor, n: torch.Tensor, fill: float = 0.0):
    n_batch, n_results = x.shape
    n = n.unsqueeze(-1)
    mask = torch.arange(n_results).repeat(n_batch, 1).type_as(x)
    x = x.float()
    x[mask >= n] = fill

    return x
