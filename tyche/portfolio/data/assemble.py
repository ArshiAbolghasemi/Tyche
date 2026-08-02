"""Align the two feature branches into dense, index-matched arrays.

Everything is keyed on ``(asset, trading_day)`` in a fixed asset order and the
shared trading-day index, so a slice ``[:, t-T+1:t+1]`` is a synchronized lookback
window across both branches and every asset. The forward H-day log return (from
daily adjusted close) is the supervised target.

The universe is resolved from the data here (see ``data.universe``) rather than
hard-coded, so the arrays are sized by whatever the price and news feeds actually
support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tyche.portfolio.config import Config
from tyche.portfolio.data.calendar import trading_days as _trading_days
from tyche.portfolio.features.daily import build_daily_features, DAILY_FEATURES
from tyche.portfolio.features.news import build_news_features, NEWS_FEATURES
from tyche.portfolio.data.loaders import load_daily, load_news_sentiment
from tyche.portfolio.data.universe import resolve_universe


@dataclass
class AlignedData:
    assets: list[str]  # length A, fixed order (== UNIVERSE)
    days: pd.DatetimeIndex  # length D
    daily: np.ndarray  # [A, D, Fd]
    news: np.ndarray  # [A, D, Fn]
    adj_close: np.ndarray  # [A, D]
    daily_names: list[str]
    news_names: list[str]

    @property
    def n_assets(self) -> int:
        return len(self.assets)


def _pivot(
    long: pd.DataFrame, cols: list[str], days: pd.DatetimeIndex, assets: list[str]
) -> np.ndarray:
    """Long ``[asset, date, *cols]`` -> array ``[A, D, len(cols)]`` (missing=NaN)."""
    arr = np.full((len(assets), len(days), len(cols)), np.nan, dtype=np.float32)
    day_pos = {d: i for i, d in enumerate(days)}
    asset_pos = {a: i for i, a in enumerate(assets)}
    for row in long.itertuples(index=False):
        ai = asset_pos.get(row.asset)
        di = day_pos.get(row.date)
        if ai is None or di is None:
            continue
        arr[ai, di] = [getattr(row, c) for c in cols]
    return arr


def assemble(cfg: Config) -> AlignedData:
    daily_raw = load_daily(cfg)
    news_raw = load_news_sentiment(cfg)

    assets = resolve_universe(daily_raw, set(news_raw["asset"].unique()), cfg.universe)
    daily_raw = daily_raw[daily_raw["asset"].isin(assets)].reset_index(drop=True)
    news_raw = news_raw[news_raw["asset"].isin(assets)].reset_index(drop=True)
    days = _trading_days(daily_raw, assets)

    daily_long = build_daily_features(daily_raw, cfg)
    news_long = build_news_features(news_raw, days, cfg)

    daily = _pivot(daily_long, DAILY_FEATURES, days, assets)
    news = _pivot(news_long, NEWS_FEATURES, days, assets)

    # Adjusted close aligned [A, D] for the forward-return target.
    adj = daily_raw.pivot(index="date", columns="asset", values="adj_close").reindex(
        index=days, columns=assets
    )
    adj_close = adj.to_numpy(dtype=np.float64).T

    return AlignedData(
        assets=assets,
        days=days,
        daily=daily,
        news=news,
        adj_close=adj_close,
        daily_names=DAILY_FEATURES,
        news_names=NEWS_FEATURES,
    )
