"""Historical return statistics from daily adjusted closes.

Shared by the Black-Litterman prior and the historical baselines. Every statistic at
decision day ``t`` uses only the trailing ``window`` of returns strictly up to ``t``
(no lookahead).
"""

from __future__ import annotations

import numpy as np


def simple_returns(adj_close: np.ndarray) -> np.ndarray:
    """[A, D] adjusted closes -> [A, D] simple daily returns (col 0 = 0)."""
    ret = np.zeros_like(adj_close)
    ret[:, 1:] = adj_close[:, 1:] / adj_close[:, :-1] - 1.0
    return ret


def trailing_window(returns: np.ndarray, t: int, window: int) -> np.ndarray:
    """Rows = assets, cols = the ``window`` daily returns ending at ``t`` inclusive."""
    lo = max(0, t - window + 1)
    return returns[:, lo : t + 1]


def historical_mean(
    returns: np.ndarray, t: int, window: int, horizon: int
) -> np.ndarray:
    """Mean daily return over the trailing window, scaled to the H-day horizon."""
    w = trailing_window(returns, t, window)
    return w.mean(axis=1) * horizon


def historical_covariance(
    returns: np.ndarray, t: int, window: int, horizon: int
) -> np.ndarray:
    """Trailing sample covariance of daily returns, scaled to the H-day horizon.
    Falls back to a diagonal if the window is too short to be full rank."""
    w = trailing_window(returns, t, window)
    if w.shape[1] < 2:
        return np.diag(np.full(w.shape[0], 1e-4))
    cov = np.cov(w) * horizon
    return cov + 1e-6 * np.eye(cov.shape[0])
