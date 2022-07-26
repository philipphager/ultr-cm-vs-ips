from typing import List

from torch import nn


def create_sequential(
    n_features: int,
    layers: List[int],
    dropouts: List[float],
    activation: nn.Module,
):
    current_size = n_features
    modules = []

    for layer, dropout in zip(layers, dropouts):
        layer = int(layer)

        modules.append(nn.Linear(current_size, layer))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))
        current_size = layer

    # Final layer always reduces to single logit
    modules.append(nn.Linear(current_size, 1))

    return nn.Sequential(*modules)
