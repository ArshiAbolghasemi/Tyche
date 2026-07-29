"""Black-Litterman posterior via PyPortfolioOpt, and its closed-form weights.

The predicted expected returns are absolute BL views (one per asset, P = I); the
predicted variances set the view uncertainty Omega, so a confident prediction (small
variance) pulls the posterior harder than an uncertain one. The prior is the
reverse-optimized equilibrium of the predicted covariance, with an equal-weight neutral
market in the absence of market caps. The posterior mean/cov come from
``pypfopt.BlackLittermanModel``; assets are addressed by integer position so the in/out
arrays stay in the pipeline's fixed universe order.

``black_litterman_weights`` is the canonical BL allocation, ``w = (delta Sigma)^-1 mu``
— which is precisely the *unconstrained* mean-variance solution. There is nowhere in
that closed form to hang a long-only bound, a position cap, or a turnover penalty, so
the weights it returns may be short and may be levered before normalization. That is
the trade being made when it is used in place of the constrained optimizer.
"""

from __future__ import annotations

import numpy as np
from pypfopt import BlackLittermanModel

from tyche.common.logging import get_logger
from tyche.portfolio.config import PortfolioConfig

log = get_logger(__name__)


def blend_covariance(
    cov_pred: np.ndarray, cov_reference: np.ndarray, shrinkage: float
) -> np.ndarray:
    """Sigma = s * predicted + (1 - s) * reference."""
    return shrinkage * cov_pred + (1.0 - shrinkage) * cov_reference


def black_litterman_posterior(
    mu_view: np.ndarray,
    cov_pred: np.ndarray,
    cov_reference: np.ndarray,
    cfg: PortfolioConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(posterior_mean, posterior_cov)`` for the N assets.

    ``mu_view`` are the predicted returns (absolute views); ``cov_pred`` supplies both
    the blended prior covariance and the per-view uncertainty (its diagonal)."""
    n = len(mu_view)
    sigma = blend_covariance(cov_pred, cov_reference, cfg.cov_shrinkage)

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


def black_litterman_weights(
    posterior_mean: np.ndarray, posterior_cov: np.ndarray, cfg: PortfolioConfig
) -> np.ndarray:
    """Closed-form BL weights ``w = (delta Sigma)^-1 mu``, normalized to sum to one.

    Solved rather than inverted, falling back to least squares on a singular system —
    with five correlated mega-caps the posterior covariance is routinely close to
    singular, which is exactly why this allocation swings hard on small changes in
    ``mu``.

    Normalizing by ``sum(w)`` is the standard convention but is unsafe when the raw
    weights sum to nearly zero: the book blows up, and a negative sum silently flips
    every position's sign. Both cases fall back to equal weight and are logged instead
    of being propagated into the backtest."""
    n = len(posterior_mean)
    a = cfg.bl_risk_aversion * posterior_cov
    try:
        raw = np.linalg.solve(a, posterior_mean)
    except np.linalg.LinAlgError:
        raw = np.linalg.lstsq(a, posterior_mean, rcond=None)[0]

    total = float(raw.sum())
    if not np.isfinite(total) or abs(total) < 1e-8 or total < 0:
        log.warning(
            "direct BL weights sum to %.2e; falling back to equal weight", total
        )
        return np.full(n, 1.0 / n)
    return raw / total
