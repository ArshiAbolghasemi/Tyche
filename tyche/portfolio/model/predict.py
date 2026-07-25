"""Generate and persist out-of-sample predictions for every rebalance date.

For each sample the model emits ``mu`` (expected forward H-day return, per asset) and
``cov`` (predictive covariance). These are the only interface between the predictive
stage and the (separate) portfolio-construction stage — saved to ``.npz`` so portfolio
tuning never re-runs the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tyche.portfolio.data.assemble import AlignedData
from tyche.portfolio.model.dataset import WindowDataset
from tyche.portfolio.model.network import MultimodalReturnModel
from tyche.portfolio.data.windows import Sample


@dataclass
class Predictions:
    decision_t: np.ndarray  # [S] trading-day index of each rebalance decision
    mu: np.ndarray  # [S, N]
    cov: np.ndarray  # [S, N, N]
    target: np.ndarray  # [S, N] realized forward H-day return (for eval)
    assets: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            decision_t=self.decision_t,
            mu=self.mu,
            cov=self.cov,
            target=self.target,
            assets=np.array(self.assets),
        )

    @staticmethod
    def load(path: Path) -> "Predictions":
        z = np.load(path, allow_pickle=True)
        return Predictions(
            z["decision_t"], z["mu"], z["cov"], z["target"], list(z["assets"])
        )


@torch.no_grad()
def predict(
    model: MultimodalReturnModel,
    data: AlignedData,
    samples: list[Sample],
    batch_size: int = 64,
) -> Predictions:
    model.eval()
    loader = DataLoader(WindowDataset(data, samples), batch_size=batch_size)
    mus, covs, tgts, ts = [], [], [], []
    for batch in loader:
        pred = model(batch["daily"], batch["news"], batch["intraday"])
        mus.append(pred.mu.numpy())
        covs.append(pred.cov.numpy())
        tgts.append(batch["target"].numpy())
        ts.append(batch["t"].numpy())
    return Predictions(
        decision_t=np.concatenate(ts),
        mu=np.concatenate(mus),
        cov=np.concatenate(covs),
        target=np.concatenate(tgts),
        assets=data.assets,
    )
