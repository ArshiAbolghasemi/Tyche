"""Raw-source loaders. Each returns a tidy, universe-filtered, UTC-indexed frame.

One responsibility only: read a parquet, normalize tickers to the canonical
universe, coerce timestamps, and hand back a clean long frame. No feature logic
here — that lives in the ``features_*`` modules.
"""

from __future__ import annotations

import pandas as pd

from tyche.portfolio.config import Config
from tyche.portfolio.data.universe import to_canonical


_NEWS_COLUMNS = ["name", "sentiment_final", "summary_text"]
# Publication-time column, in priority order: feeds that keep their own ``date``
# column are read from that; otherwise the pipeline's normalized ``valid_time``.
# Every news feature window is selected on this timestamp, so picking the wrong
# one silently shifts which articles land on which trading day.
_NEWS_TIME_COLUMNS = ["valid_time", "date"]


def _news_time_column(path) -> str:
    """The publication-time column present in the sentiment file."""
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema_arrow.names)
    for candidate in _NEWS_TIME_COLUMNS:
        if candidate in available:
            return candidate
    raise KeyError(
        f"{path} has no publication-time column — expected one of "
        f"{_NEWS_TIME_COLUMNS}, found {sorted(available)}"
    )


def load_news_sentiment(cfg: Config) -> pd.DataFrame:
    """News sentiment keyed by canonical asset and publication time.

    Every article is a row and every row carries its own score. The portfolio feature
    stage clusters near-duplicate summaries into story groups before aggregation."""
    path = cfg.paths.news_sentiment
    time_col = _news_time_column(path)
    df = pd.read_parquet(path, columns=[*_NEWS_COLUMNS, time_col])
    df["asset"] = to_canonical(df["name"])
    df = df.dropna(subset=["asset"])
    df["ts"] = pd.to_datetime(df[time_col], utc=True)
    df = df.dropna(subset=["ts"])
    keep = ["asset", "ts", "sentiment_final", "summary_text"]
    return df[keep].sort_values("ts").reset_index(drop=True)


def load_daily(cfg: Config) -> pd.DataFrame:
    """Daily OHLCV, one row per (asset, date), for every symbol in the file.

    The source is the merged long-format file produced by
    ``scripts/merge_rl2k_ohlcv.py``: ``symbol, ticker, date, open, high, low, close,
    adj_close, volume``. ``adj_close`` carries the total return; raw OHLC is kept for
    the range/gap features. Universe selection happens later (``assemble``), so this
    returns everything the file has.
    """
    df = pd.read_parquet(
        cfg.paths.daily_ohlcv,
        columns=[
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ],
    )
    df["asset"] = to_canonical(df["symbol"])
    df = df.dropna(subset=["asset", "adj_close"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    keep = ["asset", "date", "open", "high", "low", "close", "adj_close", "volume"]
    return (
        df[keep]
        .drop_duplicates(["asset", "date"])
        .sort_values(["asset", "date"])
        .reset_index(drop=True)
    )
