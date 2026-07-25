"""Representative-news feature branch — deliberately minimal.

The loader already keeps only ``is_representative`` rows and maps each to its stock,
so the news here is representative-only and per-stock. This module reduces it to the
simplest useful signal: for each ``(stock, trading-day)``, the mean neutralized
sentiment (``sentiment_final``) of that stock's news that day, plus a ``no_news``
flag. Days with no news are zeros + ``no_news = 1`` — never forward-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tyche.portfolio.config import Config

NEWS_FEATURES: list[str] = ["mean_sent", "no_news"]


def build_news_features(
    news: pd.DataFrame, trading_days: pd.DatetimeIndex, cfg: Config
) -> pd.DataFrame:
    """Return a dense long frame ``[asset, date, mean_sent, no_news]`` covering every
    (asset, trading_day) — including no-news days (``mean_sent = 0``, ``no_news = 1``).

    News published on calendar day ``d`` is attributed to trading day ``d`` (known by
    that session's close); news on a non-trading day rolls forward to the next session
    so nothing is dropped, and nothing leaks backward."""
    df = news.copy()
    cal_day = pd.DatetimeIndex(df["ts"].dt.normalize())
    # Snap each article to the first trading day >= its calendar day.
    pos = trading_days.searchsorted(cal_day, side="left")
    valid = pos < len(trading_days)
    df = df[valid].copy()
    df["date"] = trading_days[pos[valid]]

    # One number per (asset, day): mean neutralized sentiment of that stock's news.
    agg = (
        df.groupby(["asset", "date"])["sentiment_final"]
        .mean()
        .rename("mean_sent")
        .reset_index()
    )

    # Dense grid over (assets seen in news) x trading_days, so no-news days exist.
    assets = sorted(df["asset"].unique())
    grid = pd.MultiIndex.from_product(
        [assets, trading_days], names=["asset", "date"]
    ).to_frame(index=False)
    out = grid.merge(agg, on=["asset", "date"], how="left")

    out["no_news"] = out["mean_sent"].isna().astype(np.float32)
    out["mean_sent"] = out["mean_sent"].fillna(0.0)
    return out[["asset", "date", *NEWS_FEATURES]]
