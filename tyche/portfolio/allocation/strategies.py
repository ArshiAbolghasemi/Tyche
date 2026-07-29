"""Weight-generating strategies for the backtest.

Every strategy is a factory returning a ``Strategy`` closure ``t -> weights[N]``, so
the proposed model, the ablations, and the baselines all plug into the same backtest
engine with an identical universe, cost model, and rebalance schedule. The only things
that differ are how ``mu`` and ``Sigma`` are formed and how they are turned into
weights.

Two allocation routes are in play:

* **Direct Black-Litterman** — the closed form ``w = (delta Sigma)^-1 mu`` applied to
  the BL posterior. Used by the proposed strategy and by the historical-BL baseline, so
  the BL comparison is like-for-like. Unconstrained: may short, has no position cap.
* **Constrained MVO** — long-only, fully invested, per-asset capped. Used by the
  non-BL routes, which keeps ``no_bl`` a genuine ablation of the BL step.

Plus two return-free controls, ``risk_parity`` and ``hrp``, which allocate from the
covariance alone. They are the bar the model's ``mu`` has to clear.

Model-predicted moments describe **log** returns; historical moments are already
arithmetic. Predicted moments are converted before use (see ``allocation.moments``) so
both routes and the cost model operate in one consistent return space.
"""

from __future__ import annotations

import numpy as np

from tyche.portfolio.allocation.backtest import Strategy
from tyche.portfolio.allocation.black_litterman import (
    black_litterman_posterior,
    black_litterman_weights,
    blend_covariance,
)
from tyche.portfolio.allocation.moments import log_to_simple
from tyche.portfolio.config import Config
from tyche.portfolio.allocation.history import (
    historical_covariance,
    historical_mean,
    simple_returns,
)
from tyche.portfolio.allocation.optimizer import optimize_weights
from tyche.portfolio.allocation.risk_parity import (
    hierarchical_risk_parity_weights,
    risk_parity_weights,
)
from tyche.portfolio.model.predict import Predictions


def _prediction_lookup(pred: Predictions) -> dict[int, int]:
    return {int(t): i for i, t in enumerate(pred.decision_t)}


def _predicted_moments(
    pred: Predictions, i: int, cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted ``(mu, Sigma)`` for sample ``i``, in the pipeline's return space."""
    mu, cov = pred.mu[i], pred.cov[i]
    if cfg.portfolio.convert_to_simple_returns:
        mu, cov = log_to_simple(mu, cov)
    return mu, cov


def equal_weight(n_assets: int) -> Strategy:
    w = np.full(n_assets, 1.0 / n_assets)
    return lambda t: w


def risk_parity(adj_close: np.ndarray, cfg: Config) -> Strategy:
    """Equal risk contribution on the trailing historical covariance."""
    returns = simple_returns(adj_close)
    win, h = cfg.portfolio.hist_cov_window, cfg.window.holding

    def strat(t: int) -> np.ndarray:
        cov = historical_covariance(returns, t, win, h)
        return risk_parity_weights(cov, cfg.portfolio)

    return strat


def hrp(adj_close: np.ndarray, cfg: Config) -> Strategy:
    """Hierarchical risk parity on the trailing historical covariance."""
    returns = simple_returns(adj_close)
    win, h = cfg.portfolio.hist_cov_window, cfg.window.holding

    def strat(t: int) -> np.ndarray:
        cov = historical_covariance(returns, t, win, h)
        return hierarchical_risk_parity_weights(cov, cfg.portfolio)

    return strat


def historical_mvo(adj_close: np.ndarray, cfg: Config) -> Strategy:
    returns = simple_returns(adj_close)
    win, h = cfg.portfolio.hist_cov_window, cfg.window.holding

    def strat(t: int) -> np.ndarray:
        mu = historical_mean(returns, t, win, h)
        cov = historical_covariance(returns, t, win, h)
        return optimize_weights(mu, cov, cfg.portfolio)

    return strat


def historical_bl(adj_close: np.ndarray, cfg: Config) -> Strategy:
    """Historical views through BL, allocated by the same closed form as ``model_bl``."""
    returns = simple_returns(adj_close)
    win, h = cfg.portfolio.hist_cov_window, cfg.window.holding

    def strat(t: int) -> np.ndarray:
        mu = historical_mean(returns, t, win, h)
        cov = historical_covariance(returns, t, win, h)
        post_mu, post_cov = black_litterman_posterior(mu, cov, cov, cfg.portfolio)
        return black_litterman_weights(post_mu, post_cov, cfg.portfolio)

    return strat


def model_no_bl(pred: Predictions, adj_close: np.ndarray, cfg: Config) -> Strategy:
    """Predicted mu/cov straight into constrained MVO (skips Black-Litterman)."""
    lookup = _prediction_lookup(pred)
    returns = simple_returns(adj_close)
    win, h = cfg.portfolio.hist_cov_window, cfg.window.holding

    def strat(t: int) -> np.ndarray:
        i = lookup[t]
        mu, cov_pred = _predicted_moments(pred, i, cfg)
        cov_hist = historical_covariance(returns, t, win, h)
        cov = blend_covariance(cov_pred, cov_hist, cfg.portfolio.cov_shrinkage)
        return optimize_weights(mu, cov, cfg.portfolio)

    return strat


def model_bl(pred: Predictions, adj_close: np.ndarray, cfg: Config) -> Strategy:
    """The proposed strategy: predicted views -> Black-Litterman -> direct BL weights."""
    lookup = _prediction_lookup(pred)
    returns = simple_returns(adj_close)
    win, h = cfg.portfolio.hist_cov_window, cfg.window.holding

    def strat(t: int) -> np.ndarray:
        i = lookup[t]
        mu, cov_pred = _predicted_moments(pred, i, cfg)
        cov_hist = historical_covariance(returns, t, win, h)
        post_mu, post_cov = black_litterman_posterior(
            mu, cov_pred, cov_hist, cfg.portfolio
        )
        return black_litterman_weights(post_mu, post_cov, cfg.portfolio)

    return strat
