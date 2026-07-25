"""Training objective: multivariate Gaussian NLL + Huber + covariance regularizer.

The NLL is computed through the predicted Cholesky factor (never an explicit inverse
or determinant): given Sigma = L L^T, the quadratic form uses a triangular solve and
log|Sigma| = 2 * sum(log diag(L)). The Huber term stabilizes the mean early in
training; the covariance regularizer discourages exploding variances and (optionally)
covariance that changes too fast between consecutive rebalance dates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tyche.portfolio.model.network import Prediction

_LOG_2PI = 1.8378770664093453


def gaussian_nll(pred: Prediction, target: torch.Tensor) -> torch.Tensor:
    """Mean multivariate-Gaussian negative log-likelihood over the batch."""
    n = target.shape[-1]
    diff = (target - pred.mu).unsqueeze(-1)  # [B, N, 1]
    # solve L z = diff  ->  z, with quad form = ||z||^2 = diff^T Sigma^-1 diff
    z = torch.linalg.solve_triangular(pred.scale_tril, diff, upper=False)
    quad = z.pow(2).sum(dim=(-2, -1))  # [B]
    logdet = 2.0 * torch.log(torch.diagonal(pred.scale_tril, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (quad + logdet + n * _LOG_2PI).mean()


def huber(pred: Prediction, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(pred.mu, target, delta=delta)


def covariance_regularizer(pred: Prediction) -> torch.Tensor:
    """Penalize large predicted variances (trace) to keep the covariance stable."""
    var = torch.diagonal(pred.cov, dim1=-2, dim2=-1)  # [B, N]
    return var.mean()


def total_loss(
    pred: Prediction,
    target: torch.Tensor,
    huber_lambda: float,
    cov_reg_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    nll = gaussian_nll(pred, target)
    hub = huber(pred, target)
    reg = covariance_regularizer(pred)
    loss = nll + huber_lambda * hub + cov_reg_lambda * reg
    parts = {"nll": nll.item(), "huber": hub.item(), "cov_reg": reg.item()}
    return loss, parts
