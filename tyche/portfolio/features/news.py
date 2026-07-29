"""News feature branch — sentiment plus coverage intensity.

For each ``(stock, trading-day)`` this reduces that stock's articles to a small set of
causal features. Sentiment alone throws away how *loudly* a story was covered: a mean
over articles is identical whether one outlet ran it or fifty. Coverage volume and the
size a story had already accumulated are what restore that, and both are documented
predictors of subsequent drift and volatility.

Every feature is built from articles published at or before the trading day it lands
on. ``dedup_cluster_size`` counts a story's articles *up to and including* the row, so
it never reveals how big the story eventually became. Days with no news are zeros with
``no_news = 1`` — never forward-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tyche.portfolio.config import Config

NEWS_FEATURES: list[str] = [
    "mean_sent",
    "no_news",
    "sent_dispersion",
    "log_n_articles",
    "log_n_new_stories",
    "log_max_cluster_size",
]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (asset, date) summarizing that day's articles."""
    grouped = df.groupby(["asset", "date"])
    agg = grouped.agg(
        mean_sent=("sentiment_final", "mean"),
        # Disagreement across the day's stories; a single article has no dispersion.
        sent_dispersion=("sentiment_final", "std"),
        n_articles=("sentiment_final", "size"),
        # Cluster seeds landing today — stories *breaking*, as opposed to continuing.
        n_new_stories=("is_representative", "sum"),
        # How much coverage the day's largest running story had accumulated by now.
        max_cluster_size=("dedup_cluster_size", "max"),
    ).reset_index()
    agg["sent_dispersion"] = agg["sent_dispersion"].fillna(0.0)
    return agg


def build_news_features(
    news: pd.DataFrame, trading_days: pd.DatetimeIndex, cfg: Config
) -> pd.DataFrame:
    """Return a dense long frame ``[asset, date, *NEWS_FEATURES]`` covering every
    (asset, trading_day) — including no-news days.

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

    agg = _aggregate(df)

    # Dense grid over (assets seen in news) x trading_days, so no-news days exist.
    assets = sorted(df["asset"].unique())
    grid = pd.MultiIndex.from_product(
        [assets, trading_days], names=["asset", "date"]
    ).to_frame(index=False)
    out = grid.merge(agg, on=["asset", "date"], how="left")

    out["no_news"] = out["mean_sent"].isna().astype(np.float32)
    for col in ("mean_sent", "sent_dispersion", "n_articles", "n_new_stories"):
        out[col] = out[col].fillna(0.0)
    out["max_cluster_size"] = out["max_cluster_size"].fillna(0.0)

    # Counts are heavily right-skewed — a handful of days carry an order of magnitude
    # more coverage than the median — so they enter on a log scale.
    out["log_n_articles"] = np.log1p(out["n_articles"])
    out["log_n_new_stories"] = np.log1p(out["n_new_stories"])
    out["log_max_cluster_size"] = np.log1p(out["max_cluster_size"])

    return out[["asset", "date", *NEWS_FEATURES]]
