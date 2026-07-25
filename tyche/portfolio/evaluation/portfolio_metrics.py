"""Portfolio performance metrics from a backtest result.

Return/risk (cumulative gross & net, annualized return & vol, Sharpe, Sortino,
Calmar, max drawdown) and trading behaviour (average turnover, total costs, hit
rate). Daily periodicity is assumed for annualization (252 trading days).
"""

from __future__ import annotations

import numpy as np

from tyche.portfolio.allocation.backtest import BacktestResult

_ANN = 252


def _daily_returns(value: np.ndarray) -> np.ndarray:
    return value[1:] / value[:-1] - 1.0


def max_drawdown(value: np.ndarray) -> float:
    peak = np.maximum.accumulate(value)
    return float((value / peak - 1.0).min())


def evaluate_portfolio(
    result: BacktestResult, gross_value: np.ndarray
) -> dict[str, float]:
    """``result.value`` is net of costs; ``gross_value`` is the same path without
    the per-rebalance cost charges, for the gross-vs-net comparison."""
    r = _daily_returns(result.value)
    ann_ret = float(np.mean(r) * _ANN)
    ann_vol = float(np.std(r) * np.sqrt(_ANN))
    downside = r[r < 0]
    sortino_vol = float(np.std(downside) * np.sqrt(_ANN)) if downside.size else np.nan
    mdd = max_drawdown(result.value)

    return {
        "cum_return_gross": float(gross_value[-1] - 1.0),
        "cum_return_net": float(result.value[-1] - 1.0),
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "sortino": ann_ret / sortino_vol if sortino_vol and sortino_vol > 0 else np.nan,
        "calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
        "max_drawdown": mdd,
        "avg_turnover": float(np.mean(result.turnover))
        if result.turnover.size
        else 0.0,
        "total_costs": float(np.sum(result.costs)),
        "hit_rate": float(np.mean(r > 0)),
    }
