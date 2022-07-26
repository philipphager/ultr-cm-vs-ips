import sys

import torch


def print_memory_footprint(x: torch.Tensor, name: str):
    print(f"Memory footprint of {name}: {sys.getsizeof(x.storage()) / 1e+6:.2f}mb")
