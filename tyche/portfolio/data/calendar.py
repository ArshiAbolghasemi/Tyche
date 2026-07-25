"""The trading-day index.

The daily OHLCV frame defines the sessions the whole pipeline runs on: every
feature, label, split, and rebalance date snaps to this ordered index of trading
days. Derived once from the intersection of dates all universe assets share, so a
sample at day ``t`` always has every asset present.
"""

from __future__ import annotations

import pandas as pd

from tyche.portfolio.data.universe import UNIVERSE


def trading_days(daily: pd.DataFrame) -> pd.DatetimeIndex:
    """Ordered trading days on which *every* universe asset has a daily bar."""
    counts = daily.groupby("date")["asset"].nunique()
    full = counts[counts >= len(UNIVERSE)].index
    return pd.DatetimeIndex(sorted(full))
