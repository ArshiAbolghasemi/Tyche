"""Portfolio performance metrics from a backtest result.

Return/risk (cumulative gross & net, annualized return & vol, Sharpe, Sortino,
Calmar, max drawdown), trading behaviour (average turnover, total costs, hit
rate), and **exposure** (gross, net, leverage). Daily periodicity is assumed for
annualization (252 trading days).

Exposure is reported unconditionally because a return number is meaningless without
it. Only ``MVO`` is genuinely constrained (long-only, fully invested); the direct
Black-Litterman weights are the *unconstrained* mean-variance solution normalized by
**net** exposure, which leaves gross exposure mathematically unbounded — a book can
report a tidy 100% net while actually running 300%, 500% or worse long/short. A
strategy that reaches a high return by quietly levering up is not the same strategy
as one that reaches it fully invested, and without these columns the two are
indistinguishable in the results table.

Note the two senses of "gross" in this module, which are unrelated:
``cum_return_gross`` is gross *of transaction costs*; ``gross_exposure`` is the sum
of absolute weights.
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


def exposure_metrics(weights: np.ndarray) -> dict[str, float]:
    """Gross / net / short exposure and leverage from the per-rebalance weights.

    * **gross** ``sum |w_i|`` — total capital at risk, long plus short.
    * **net** ``sum w_i`` — directional market exposure.
    * **short** ``sum |w_i|`` over short positions only.
    * **leverage** — the *worst* gross exposure over the backtest, which is the
      number that determines whether the book was financeable at all. The average
      hides a single catastrophic rebalance.

    A long-only fully-invested book reads gross 1.00 / net 1.00 / short 0.00 /
    leverage 1.00; anything above that is borrowing, and the returns must be read
    against it.
    """
    if weights.size == 0:
        return {
            "gross_exposure": float("nan"),
            "net_exposure": float("nan"),
            "short_exposure": float("nan"),
            "leverage": float("nan"),
        }
    w = np.atleast_2d(weights)
    gross = np.abs(w).sum(axis=1)
    net = w.sum(axis=1)
    short = np.where(w < 0, -w, 0.0).sum(axis=1)
    return {
        "gross_exposure": float(gross.mean()),
        "net_exposure": float(net.mean()),
        "short_exposure": float(short.mean()),
        "leverage": float(gross.max()),
    }


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
        **exposure_metrics(result.weights),
    }
