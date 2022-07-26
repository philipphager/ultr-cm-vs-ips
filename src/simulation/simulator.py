import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.baseline import LightGBMRanker
from src.data.preprocessing.convert import RatingDataset
from src.simulation.user import UserModel


class ClickDataset(Dataset):
    def __init__(
        self,
        q: torch.Tensor,
        n: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        query_ids: torch.Tensor,
        y_clicks: torch.Tensor,
    ):
        self.q = q
        self.n = n
        self.x = x
        self.y = y
        self.query_ids = query_ids
        self.y_clicks = y_clicks

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, i: int):
        query_id = self.query_ids[i]

        q = self.q[query_id]
        n = self.n[query_id]
        x = self.x[query_id]
        y = self.y[query_id]
        y_click = self.y_clicks[i].float()

        return q, n, x, y, y_click


class Simulator:
    def __init__(self, baseline_model: LightGBMRanker, user_model: UserModel):
        self.baseline_model = baseline_model
        self.user_model = user_model

    def __call__(
        self, dataset: RatingDataset, n_sessions: int, aggregate: bool = False
    ):
        dataset = self.rank(dataset)

        q, n, x, y = dataset[:]
        n_queries = len(q)

        probabilities = self.user_model(y)

        query_ids = []
        y_clicks = []

        print(f"Generating {n_sessions}, aggregating CTRs: {aggregate}")
        sessions_per_query = torch.randint(n_queries, (n_sessions,))
        sessions_per_query = torch.bincount(sessions_per_query, minlength=n_queries)

        for i in tqdm(range(n_queries), f"Generating {n_sessions} sessions"):
            if sessions_per_query[i] == 0:
                continue

            # Sample all clicks for a given query
            query_probabilities = probabilities[i].repeat(sessions_per_query[i], 1)
            clicks = torch.bernoulli(query_probabilities)

            if aggregate:
                clicks = clicks.mean(0).unsqueeze(0)
                query_id = torch.tensor([i])
            else:
                # Convert to boolean mask
                clicks = clicks == 1
                query_id = torch.full((sessions_per_query[i],), i)

            query_ids.append(query_id)
            y_clicks.append(clicks)

        query_ids = torch.cat(query_ids, dim=0)
        y_clicks = torch.cat(y_clicks, dim=0)

        return ClickDataset(q, n, x, y, query_ids, y_clicks)

    def rank(self, dataset: RatingDataset):
        q, n, x, y = dataset[:]
        y_predict = self.baseline_model.predict(dataset)

        n_batch, n_results, n_features = x.shape
        idx = torch.argsort(y_predict, dim=1, descending=True)
        x = torch.gather(x, 1, idx.unsqueeze(-1).repeat(1, 1, n_features))
        y = torch.gather(y, 1, idx)
        return RatingDataset(q, n, x, y)
