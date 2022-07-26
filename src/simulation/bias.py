import torch


def get_position_bias(n_results: int, strength: float = 1.0):
    """
    Simulate position bias based on Joachims 2017:
    https://www.cs.cornell.edu/people/tj/publications/joachims_etal_17a.pdf

    I.e. position bias at each rank follows: (1 / rank) ** strength

    Args:
        n_results: Number of documents per query
        strength: Strength of position bias
            > 1.0 Increase bias, decrease examination probability at low ranks
            < 1.0 Decrease bias, increase examination probability at low ranks
            0 No position bias, all documents are observed by user

    Returns: Tensor of size (n_results,) with the position bias for each rank.

    """
    return 1 / (torch.arange(n_results) + 1) ** strength
