import pandas as pd
import torch

from pytorchltr.evaluation import arp, ndcg


def get_metrics(
    y_predict: torch.Tensor,
    y_true: torch.Tensor,
    n: torch.Tensor,
    q: torch.Tensor,
    stage: str = "",
):
    return pd.DataFrame(
        {
            "query": q,
            "stage": stage,
            "arp": arp(y_predict, y_true, n).detach().numpy(),
            "ndcg@1": ndcg(y_predict, y_true, n, k=1).detach().numpy(),
            "ndcg@5": ndcg(y_predict, y_true, n, k=5).detach().numpy(),
            "ndcg@10": ndcg(y_predict, y_true, n, k=10).detach().numpy(),
            "ndcg": ndcg(y_predict, y_true, n).mean().detach().numpy(),
        }
    )
