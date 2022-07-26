import torch

from pytorchltr.evaluation import arp, ndcg


def get_metrics(
    y_predict: torch.Tensor, y_true: torch.Tensor, n: torch.Tensor, prefix: str = ""
):
    return {
        f"{prefix}arp": arp(y_predict, y_true, n).mean().detach(),
        f"{prefix}ndcg@1": ndcg(y_predict, y_true, n, k=1).mean().detach(),
        f"{prefix}ndcg@5": ndcg(y_predict, y_true, n, k=5).mean().detach(),
        f"{prefix}ndcg@10": ndcg(y_predict, y_true, n, k=10).mean().detach(),
        f"{prefix}ndcg": ndcg(y_predict, y_true, n).mean().detach(),
    }
