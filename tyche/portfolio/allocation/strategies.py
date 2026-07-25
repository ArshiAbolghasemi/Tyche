"""Weight-generating strategies for the backtest.

Every strategy is a factory returning a ``Strategy`` closure ``t -> weights[N]``, so
the proposed model, the ablations, and the classical baselines all plug into the same
backtest engine with an identical universe, constraints, and rebalance schedule. The
only thing that differs is how ``mu`` and ``cov`` are formed before optimization.
"""

from __future__ import annotations

import numpy as np

from tyche.portfolio.allocation.backtest import Strategy
from tyche.portfolio.allocation.black_litterman import (
    black_litterman_posterior,
    blend_covariance,
)
from tyche.portfolio.config import Config
from tyche.portfolio.allocation.history import (
    historical_covariance,
    historical_mean,
    simple_returns,
)
from tyche.portfolio.allocation.optimizer import optimize_weights
from tyche.portfolio.model.predict import Predictions


def _prediction_lookup(pred: Predictions) -> dict[int, int]:
    return {int(t): i for i, t in enumerate(pred.decision_t)}


def equal_weight(n_assets: int) -> Strategy:
    w = np.full(n_assets, 1.0 / n_assets)
    return lambda t: w


def historical_mvo(adj_close: np.ndarray, cfg: Config) -> Strategy:
    returns = simple_returns(adj_close)
    win, h = cfg.portfolio.hist_cov_window, cfg.window.holding

    def strat(t: int) -> np.ndarray:
        mu = historical_mean(returns, t, win, h)
        cov = historical_covariance(returns, t, win, h)
        return optimize_weights(mu, cov, cfg.portfolio)

    return strat


def historical_bl(adj_close: np.ndarray, cfg: Config) -> Strategy:
    returns = simple_returns(adj_close)
    win, h = cfg.portfolio.hist_cov_window, cfg.window.holding

    def strat(t: int) -> np.ndarray:
        mu = historical_mean(returns, t, win, h)
        cov = historical_covariance(returns, t, win, h)
        post_mu, post_cov = black_litterman_posterior(mu, cov, cov, cfg.portfolio)
        return optimize_weights(post_mu, post_cov, cfg.portfolio)

    return strat


def model_no_bl(pred: Predictions, adj_close: np.ndarray, cfg: Config) -> Strategy:
    """Predicted mu/cov straight into MVO (skips the Black-Litterman blend)."""
    lookup = _prediction_lookup(pred)
    returns = simple_returns(adj_close)
    win, h = cfg.portfolio.hist_cov_window, cfg.window.holding

    def strat(t: int) -> np.ndarray:
        i = lookup[t]
        cov_hist = historical_covariance(returns, t, win, h)
        cov = blend_covariance(pred.cov[i], cov_hist, cfg.portfolio.cov_shrinkage)
        return optimize_weights(pred.mu[i], cov, cfg.portfolio)

    return strat


def model_bl(pred: Predictions, adj_close: np.ndarray, cfg: Config) -> Strategy:
    """The proposed strategy: predicted views -> Black-Litterman -> constrained MVO."""
    lookup = _prediction_lookup(pred)
    returns = simple_returns(adj_close)
    win, h = cfg.portfolio.hist_cov_window, cfg.window.holding

    def strat(t: int) -> np.ndarray:
        i = lookup[t]
        cov_hist = historical_covariance(returns, t, win, h)
        post_mu, post_cov = black_litterman_posterior(
            pred.mu[i], pred.cov[i], cov_hist, cfg.portfolio
        )
        return optimize_weights(post_mu, post_cov, cfg.portfolio)

    return strat
