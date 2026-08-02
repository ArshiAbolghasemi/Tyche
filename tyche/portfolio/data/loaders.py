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
_NEWS_TIME_COLUMNS = ["date", "valid_time"]


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
    """Daily OHLCV, one row per (asset, date). Uses ``Adjusted_close`` for total
    return and keeps raw OHLC for range/gap features."""
    df = pd.read_parquet(
        cfg.paths.daily_ohlcv,
        columns=[
            "ticker",
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adjusted_close",
            "Volume",
        ],
    )
    df["asset"] = to_canonical(df["ticker"])
    df = df.dropna(subset=["asset"])
    df["date"] = pd.to_datetime(df["Date"], utc=True).dt.normalize()
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adjusted_close": "adj_close",
            "Volume": "volume",
        }
    )
    keep = ["asset", "date", "open", "high", "low", "close", "adj_close", "volume"]
    return df[keep].sort_values(["asset", "date"]).reset_index(drop=True)


def load_intraday(cfg: Config) -> pd.DataFrame:
    """One-minute OHLCV bars, one row per (asset, datetime). Restricted to the
    regular session so time-of-day features are comparable across days."""
    df = pd.read_parquet(
        cfg.paths.intraday_ohlcv,
        columns=["ticker", "Datetime", "Open", "High", "Low", "Close", "Volume"],
    )
    df["asset"] = to_canonical(df["ticker"])
    df = df.dropna(subset=["asset"])
    dt = pd.to_datetime(df["Datetime"], utc=True)
    df["dt"] = dt
    df["date"] = dt.dt.normalize()
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    session = df["dt"].dt.tz_convert("America/New_York").dt.time
    start = pd.to_datetime(cfg.intraday.session_start).time()
    end = pd.to_datetime(cfg.intraday.session_end).time()
    df = df[(session >= start) & (session <= end)]
    keep = ["asset", "dt", "date", "open", "high", "low", "close", "volume"]
    return df[keep].sort_values(["asset", "dt"]).reset_index(drop=True)
