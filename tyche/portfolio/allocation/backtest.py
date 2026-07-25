"""Event-driven backtest with transaction costs and slippage.

Rebalances every ``H`` trading days: at each rebalance the strategy produces target
weights, costs are charged on the turnover versus the drifted current weights, and
the portfolio is held (weights drifting with returns) until the next rebalance. Emits
a daily portfolio-value series plus per-rebalance turnover/cost bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from tyche.portfolio.config import Config
from tyche.portfolio.allocation.history import simple_returns

# A strategy maps a rebalance day index -> target weight vector [N].
Strategy = Callable[[int], np.ndarray]


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    value: np.ndarray  # daily portfolio value (starts at 1.0)
    weights: np.ndarray  # [n_rebal, N] target weights per rebalance
    rebal_t: np.ndarray  # [n_rebal] decision-day indices
    turnover: np.ndarray  # [n_rebal] L1 turnover per rebalance
    costs: np.ndarray  # [n_rebal] fractional cost charged per rebalance


def _drift(weights: np.ndarray, day_ret: np.ndarray) -> np.ndarray:
    grown = weights * (1.0 + day_ret)
    total = grown.sum()
    return grown / total if total > 0 else weights


def run_backtest(
    adj_close: np.ndarray,
    days: pd.DatetimeIndex,
    rebalance_t: list[int],
    strategy: Strategy,
    cfg: Config,
) -> BacktestResult:
    returns = simple_returns(adj_close)
    cost_rate = (cfg.portfolio.transaction_cost_bps + cfg.portfolio.slippage_bps) / 1e4
    n = adj_close.shape[0]

    start, end = rebalance_t[0], rebalance_t[-1] + cfg.window.holding
    end = min(end, adj_close.shape[1] - 1)

    value = [1.0]
    cur_val = 1.0
    weights = np.zeros(n)
    rebal_set = set(rebalance_t)

    w_hist, to_hist, cost_hist = [], [], []
    for t in range(start, end + 1):
        if t in rebal_set:
            target = strategy(t)
            turnover = np.abs(target - weights).sum()
            charge = turnover * cost_rate
            cur_val *= 1.0 - charge
            weights = target
            w_hist.append(target)
            to_hist.append(turnover)
            cost_hist.append(charge)
        if t + 1 <= end:
            day_ret = returns[:, t + 1]
            cur_val *= 1.0 + float(weights @ day_ret)
            weights = _drift(weights, day_ret)
            value.append(cur_val)

    return BacktestResult(
        dates=days[start : start + len(value)],
        value=np.array(value),
        weights=np.array(w_hist),
        rebal_t=np.array(sorted(rebal_set)),
        turnover=np.array(to_hist),
        costs=np.array(cost_hist),
    )
