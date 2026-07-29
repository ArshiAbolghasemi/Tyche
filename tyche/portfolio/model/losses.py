"""Training objective: distributional NLL + Huber + covariance regularizer.

The NLL can be multivariate Gaussian or multivariate Student-t. It is computed through
the predicted Cholesky factor (never an explicit inverse or determinant): given
Sigma = L L^T, the quadratic form uses a triangular solve and log|Sigma| =
2 * sum(log diag(L)). The Huber term stabilizes the mean early in training; the
covariance regularizer discourages exploding variances.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tyche.portfolio.model.network import Prediction

_LOG_2PI = 1.8378770664093453
_LOG_PI = 1.1447298858494002


def _normalized_distribution(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    if normalized in {"gaussian", "normal"}:
        return "gaussian"
    if normalized in {"student_t", "t_student", "student", "t"}:
        return "student_t"
    raise ValueError(
        f"target_distribution must be 'gaussian' or 'student_t', got {name!r}"
    )


def gaussian_nll(pred: Prediction, target: torch.Tensor) -> torch.Tensor:
    """Mean multivariate-Gaussian negative log-likelihood over the batch."""
    n = target.shape[-1]
    diff = (target - pred.mu).unsqueeze(-1)  # [B, N, 1]
    # solve L z = diff  ->  z, with quad form = ||z||^2 = diff^T Sigma^-1 diff
    z = torch.linalg.solve_triangular(pred.scale_tril, diff, upper=False)
    quad = z.pow(2).sum(dim=(-2, -1))  # [B]
    logdet = 2.0 * torch.log(torch.diagonal(pred.scale_tril, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (quad + logdet + n * _LOG_2PI).mean()


def student_t_nll(
    pred: Prediction, target: torch.Tensor, degrees_of_freedom: float
) -> torch.Tensor:
    """Mean multivariate Student-t NLL over the batch.

    ``pred.cov`` is interpreted as the covariance matrix. For Student-t with
    ``nu > 2``, covariance is ``nu / (nu - 2) * scale``; therefore the likelihood uses
    ``scale = cov * (nu - 2) / nu`` so the model's covariance head keeps the same
    meaning under both supported target distributions.
    """
    if degrees_of_freedom <= 2.0:
        raise ValueError("student_t_df must be > 2 so covariance is finite")
    n = target.shape[-1]
    nu = torch.as_tensor(
        float(degrees_of_freedom), dtype=target.dtype, device=target.device
    )
    scale_factor = torch.sqrt((nu - 2.0) / nu)
    scale_tril = pred.scale_tril * scale_factor
    diff = (target - pred.mu).unsqueeze(-1)
    z = torch.linalg.solve_triangular(scale_tril, diff, upper=False)
    quad = z.pow(2).sum(dim=(-2, -1))
    logdet = 2.0 * torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(-1)
    log_norm = (
        torch.lgamma((nu + n) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * (n * torch.log(nu) + n * _LOG_PI + logdet)
    )
    log_kernel = -0.5 * (nu + n) * torch.log1p(quad / nu)
    return -(log_norm + log_kernel).mean()


def distribution_nll(
    pred: Prediction,
    target: torch.Tensor,
    distribution: str = "gaussian",
    student_t_df: float = 5.0,
) -> torch.Tensor:
    """Mean NLL under the configured predictive target distribution."""
    match _normalized_distribution(distribution):
        case "gaussian":
            return gaussian_nll(pred, target)
        case "student_t":
            return student_t_nll(pred, target, student_t_df)
    raise AssertionError("unreachable distribution branch")


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
    target_distribution: str = "gaussian",
    student_t_df: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    nll = distribution_nll(pred, target, target_distribution, student_t_df)
    hub = huber(pred, target)
    reg = covariance_regularizer(pred)
    loss = nll + huber_lambda * hub + cov_reg_lambda * reg
    parts = {"nll": nll.item(), "huber": hub.item(), "cov_reg": reg.item()}
    return loss, parts
