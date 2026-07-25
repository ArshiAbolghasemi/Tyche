"""Train the predictive model; select the checkpoint by validation NLL.

Pure model training — no portfolio objective ever enters here (per the spec, the
network optimizes only the return-distribution likelihood). Returns the best model
(lowest validation NLL) plus the per-epoch history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from tyche.portfolio.data.assemble import AlignedData
from tyche.portfolio.config import Config
from tyche.portfolio.model.dataset import WindowDataset
from tyche.portfolio.model.losses import gaussian_nll, total_loss
from tyche.portfolio.model.network import MultimodalReturnModel
from tyche.portfolio.data.windows import Sample


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainResult:
    model: MultimodalReturnModel
    best_val_nll: float
    history: list[dict]


def build_model(data: AlignedData, cfg: Config, use_news=True, use_intraday=True):
    return MultimodalReturnModel(
        n_assets=data.n_assets,
        daily_features=len(data.daily_names),
        news_features=len(data.news_names),
        intraday_features=len(data.intraday_names),
        cfg=cfg.model,
        use_news=use_news,
        use_intraday=use_intraday,
    )


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def _val_nll(model, loader, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = _move(batch, device)
        pred = model(batch["daily"], batch["news"], batch["intraday"])
        total += gaussian_nll(pred, batch["target"]).item() * len(batch["target"])
        count += len(batch["target"])
    return total / max(count, 1)


def train_model(
    data: AlignedData,
    train_samples: list[Sample],
    val_samples: list[Sample],
    cfg: Config,
    use_news=True,
    use_intraday=True,
) -> TrainResult:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    model = build_model(data, cfg, use_news, use_intraday).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    train_loader = DataLoader(
        WindowDataset(data, train_samples),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        WindowDataset(data, val_samples), batch_size=cfg.train.batch_size
    )

    best_state, best_nll, history, stale = None, float("inf"), [], 0
    for epoch in range(cfg.train.epochs):
        model.train()
        running = {"nll": 0.0, "huber": 0.0, "cov_reg": 0.0}
        for batch in train_loader:
            batch = _move(batch, device)
            opt.zero_grad()
            pred = model(batch["daily"], batch["news"], batch["intraday"])
            loss, parts = total_loss(
                pred, batch["target"], cfg.train.huber_lambda, cfg.train.cov_reg_lambda
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            for k in running:
                running[k] += parts[k]

        val_nll = _val_nll(model, val_loader, device) if val_samples else float("nan")
        history.append(
            {
                "epoch": epoch,
                "val_nll": val_nll,
                **{k: v / len(train_loader) for k, v in running.items()},
            }
        )

        if val_samples and val_nll < best_nll - 1e-4:
            best_nll, best_state, stale = val_nll, _clone(model), 0
        else:
            stale += 1
            if stale >= cfg.train.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return TrainResult(model=model, best_val_nll=best_nll, history=history)


def _clone(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
