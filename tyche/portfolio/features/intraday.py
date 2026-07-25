"""Intraday OHLCV feature branch.

One-minute bars are resampled to a coarser interval (15-minute default), turned
into normalized per-bar features, and returned as a dense tensor of shape
``(asset, date, bars_per_day, n_features)``. The model's intraday encoder later
compresses the ``bars_per_day`` axis into one embedding per trading day. Raw prices
are avoided; features are returns, ranges, ratios, and a time-of-day position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tyche.portfolio.config import Config

INTRADAY_FEATURES: list[str] = [
    "log_ret",
    "oc_ret",
    "hl_range",
    "realized_vol",
    "log_volume",
    "rel_volume",
    "price_to_vwap",
    "cum_ret",
    "roll_vol",
    "tod",
]


def _resample_day(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample one asset-day of 1-min bars to ``rule`` OHLCV."""
    g = bars.set_index("dt").resample(rule, label="left", closed="left")
    out = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"])
    return out


def _bar_features(day: pd.DataFrame) -> pd.DataFrame:
    close, vol = day["close"], day["volume"]
    typical = (day["high"] + day["low"] + close) / 3.0
    cum_vol = vol.cumsum().replace(0, np.nan)
    vwap = (typical * vol).cumsum() / cum_vol
    log_ret = np.log(close).diff()

    out = pd.DataFrame(index=day.index)
    out["log_ret"] = log_ret
    out["oc_ret"] = np.log(close / day["open"])
    out["hl_range"] = (day["high"] - day["low"]) / close
    out["realized_vol"] = log_ret.pow(2).rolling(4, min_periods=1).sum().pow(0.5)
    out["log_volume"] = np.log1p(vol)
    out["rel_volume"] = vol / vol.expanding().mean()
    out["price_to_vwap"] = close / vwap - 1.0
    out["cum_ret"] = np.log(close / close.iloc[0])
    out["roll_vol"] = log_ret.rolling(4, min_periods=1).std()
    out["tod"] = np.linspace(0.0, 1.0, len(day))  # normalized time-of-day position
    return out.fillna(0.0)


def build_intraday_tensor(
    intraday: pd.DataFrame, cfg: Config
) -> tuple[np.ndarray, dict[tuple[str, pd.Timestamp], int]]:
    """Return ``(tensor, index)`` where ``tensor`` has shape
    ``(n_asset_days, max_bars, n_features)`` and ``index`` maps
    ``(asset, date) -> row``. Days are right-padded with zeros to ``max_bars``."""
    max_bars = cfg.intraday.max_bars_per_day
    n_feat = len(INTRADAY_FEATURES)
    rows: list[np.ndarray] = []
    index: dict[tuple[str, pd.Timestamp], int] = {}

    for (asset, date), bars in intraday.groupby(["asset", "date"], sort=True):
        res = _resample_day(bars, cfg.intraday.resample)
        if res.empty:
            continue
        feats = _bar_features(res)[INTRADAY_FEATURES].to_numpy(dtype=np.float32)
        feats = feats[:max_bars]
        padded = np.zeros((max_bars, n_feat), dtype=np.float32)
        padded[: len(feats)] = feats
        index[(asset, pd.Timestamp(date))] = len(rows)
        rows.append(padded)

    tensor = (
        np.stack(rows) if rows else np.zeros((0, max_bars, n_feat), dtype=np.float32)
    )
    return tensor, index
