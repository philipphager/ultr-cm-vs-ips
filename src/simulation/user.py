from abc import ABC, abstractmethod

import torch

from src.simulation.bias import get_position_bias


class UserModel(ABC):
    @abstractmethod
    def __call__(self, y: torch.Tensor):
        pass


class GradedPBM(UserModel):
    def __init__(self, position_bias: float, noise: float):
        self.position_bias = position_bias
        self.noise = noise

    def __call__(self, y: torch.Tensor):
        n_queries, n_results = y.shape

        relevance = self.noise + (1 - self.noise) * (2**y - 1) / (2**4 - 1)
        examination = get_position_bias(n_results, self.position_bias)
        examination = examination.repeat(n_queries, 1)

        return relevance * examination


class BinaryPBM(UserModel):
    def __init__(self, position_bias: float, noise: float):
        self.position_bias = position_bias
        self.noise = noise

    def __call__(self, y: torch.Tensor):
        n_queries, n_results = y.shape

        relevance = (y > 2).float().clip(min=self.noise)
        examination = get_position_bias(n_results, self.position_bias)
        examination = examination.repeat(n_queries, 1)

        return relevance * examination
