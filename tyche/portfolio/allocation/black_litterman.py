"""Black-Litterman posterior via PyPortfolioOpt.

The predicted expected returns are absolute BL views (one per asset, P = I); the
predicted variances set the view uncertainty Omega, so a confident prediction (small
variance) pulls the posterior harder than an uncertain one. The prior is the
reverse-optimized equilibrium of a blended covariance (predicted shrunk toward
historical), with an equal-weight neutral market in the absence of market caps. The
posterior mean/cov come from ``pypfopt.BlackLittermanModel``; assets are addressed by
integer position so the in/out arrays stay in the pipeline's fixed universe order.
"""

from __future__ import annotations

import numpy as np
from pypfopt import BlackLittermanModel

from tyche.portfolio.config import PortfolioConfig


def blend_covariance(
    cov_pred: np.ndarray, cov_hist: np.ndarray, shrinkage: float
) -> np.ndarray:
    """Sigma = s * predicted + (1 - s) * historical."""
    return shrinkage * cov_pred + (1.0 - shrinkage) * cov_hist


def black_litterman_posterior(
    mu_view: np.ndarray,
    cov_pred: np.ndarray,
    cov_hist: np.ndarray,
    cfg: PortfolioConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(posterior_mean, posterior_cov)`` for the N assets.

    ``mu_view`` are the predicted returns (absolute views); ``cov_pred`` supplies both
    the blended prior covariance and the per-view uncertainty (its diagonal)."""
    n = len(mu_view)
    sigma = blend_covariance(cov_pred, cov_hist, cfg.cov_shrinkage)

    w_eq = np.full(n, 1.0 / n)
    pi = cfg.bl_risk_aversion * sigma @ w_eq  # equilibrium prior mean
    omega = np.diag(np.clip(np.diag(cov_pred), 1e-8, None))  # view uncertainty
    views = {i: float(mu_view[i]) for i in range(n)}  # absolute view per asset (P = I)

    bl = BlackLittermanModel(
        sigma, pi=pi, absolute_views=views, omega=omega, tau=cfg.bl_tau
    )
    posterior_mean = np.asarray(bl.bl_returns()).reshape(-1)
    posterior_cov = np.asarray(bl.bl_cov())
    return posterior_mean, posterior_cov
