"""Align the three feature branches into dense, index-matched arrays.

Everything is keyed on ``(asset, trading_day)`` in a fixed asset order and the
shared trading-day index, so a slice ``[:, t-T+1:t+1]`` is a synchronized lookback
window across all branches and assets. The forward H-day log return (from daily
adjusted close) is the supervised target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tyche.portfolio.config import Config
from tyche.portfolio.data.calendar import trading_days as _trading_days
from tyche.portfolio.features.daily import build_daily_features, DAILY_FEATURES
from tyche.portfolio.features.intraday import build_intraday_tensor, INTRADAY_FEATURES
from tyche.portfolio.features.news import build_news_features, NEWS_FEATURES
from tyche.portfolio.data.loaders import load_daily, load_intraday, load_news_sentiment
from tyche.portfolio.data.universe import UNIVERSE


@dataclass
class AlignedData:
    assets: list[str]  # length A, fixed order (== UNIVERSE)
    days: pd.DatetimeIndex  # length D
    daily: np.ndarray  # [A, D, Fd]
    news: np.ndarray  # [A, D, Fn]
    intraday: np.ndarray  # [A, D, B, Fi]
    adj_close: np.ndarray  # [A, D]
    daily_names: list[str]
    news_names: list[str]
    intraday_names: list[str]

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
    days = _trading_days(daily_raw)
    assets = UNIVERSE

    daily_long = build_daily_features(daily_raw, cfg)
    news_long = build_news_features(load_news_sentiment(cfg), days, cfg)
    intra_tensor, intra_idx = build_intraday_tensor(load_intraday(cfg), cfg)

    daily = _pivot(daily_long, DAILY_FEATURES, days, assets)
    news = _pivot(news_long, NEWS_FEATURES, days, assets)

    # Intraday: scatter the (asset-day, bars, feat) tensor into [A, D, B, Fi].
    B, Fi = intra_tensor.shape[1], intra_tensor.shape[2]
    intraday = np.zeros((len(assets), len(days), B, Fi), dtype=np.float32)
    asset_pos = {a: i for i, a in enumerate(assets)}
    day_pos = {d: i for i, d in enumerate(days)}
    for (asset, date), row in intra_idx.items():
        ai, di = asset_pos.get(asset), day_pos.get(pd.Timestamp(date))
        if ai is not None and di is not None:
            intraday[ai, di] = intra_tensor[row]

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
        intraday=intraday,
        adj_close=adj_close,
        daily_names=DAILY_FEATURES,
        news_names=NEWS_FEATURES,
        intraday_names=INTRADAY_FEATURES,
    )
