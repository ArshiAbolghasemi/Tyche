"""Weight-generating strategies for the portfolio backtest.

Each factory returns a ``Strategy`` closure ``t -> weights[N]``. The public model set is
restricted to BL, MVO, RP, and HRP, all using trailing historical moments so they share
one universe, cost model, and rebalance schedule.
"""

from __future__ import annotations

import numpy as np

from tyche.portfolio.allocation.backtest import Strategy
from tyche.portfolio.allocation.black_litterman import (
    black_litterman_posterior,
    black_litterman_weights,
)
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


def rp(adj_close: np.ndarray, cfg: Config) -> Strategy:
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


def mvo(adj_close: np.ndarray, cfg: Config) -> Strategy:
    """Constrained MVO on trailing historical mean and covariance."""
    returns = simple_returns(adj_close)
    win, h = cfg.portfolio.hist_cov_window, cfg.window.holding

    def strat(t: int) -> np.ndarray:
        mu = historical_mean(returns, t, win, h)
        cov = historical_covariance(returns, t, win, h)
        return optimize_weights(mu, cov, cfg.portfolio)

    return strat


def bl(adj_close: np.ndarray, cfg: Config) -> Strategy:
    """Historical views through BL, allocated by the direct BL closed form."""
    returns = simple_returns(adj_close)
    win, h = cfg.portfolio.hist_cov_window, cfg.window.holding

    def strat(t: int) -> np.ndarray:
        mu = historical_mean(returns, t, win, h)
        cov = historical_covariance(returns, t, win, h)
        post_mu, post_cov = black_litterman_posterior(mu, cov, cov, cfg.portfolio)
        return black_litterman_weights(post_mu, post_cov, cfg.portfolio)

    return strat
