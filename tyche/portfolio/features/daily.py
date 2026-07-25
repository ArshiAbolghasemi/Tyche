"""Daily OHLCV feature branch.

One row per (asset, date) of normalized, strictly causal daily features: every
rolling statistic at day ``d`` uses only observations up to and including ``d``
(known at that day's close), so nothing leaks from the forward target window.
Raw prices are avoided in favor of returns, ranges, and ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tyche.portfolio.config import Config

# Feature columns produced, in a fixed order (the model relies on this width).
DAILY_FEATURES: list[str] = [
    "ret_close",
    "ret_oc",
    "ret_overnight",
    "ret_adj",
    "hl_range",
    "roll_vol",
    "roll_mean_ret",
    "rel_volume",
    "vol_change",
    "amihud",
    "mom_5",
    "mom_20",
    "rsi",
    "atr",
    "z_ret",
    "z_vol",
    "ma_ratio",
]


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean() / df["close"]


def _per_asset(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = cfg.daily
    close, adj, vol = df["close"], df["adj_close"], df["volume"]
    out = pd.DataFrame(index=df.index)

    ret_adj = np.log(adj).diff()
    out["ret_close"] = np.log(close).diff()
    out["ret_oc"] = np.log(close / df["open"])
    out["ret_overnight"] = np.log(df["open"] / close.shift(1))
    out["ret_adj"] = ret_adj
    out["hl_range"] = (df["high"] - df["low"]) / close

    out["roll_vol"] = ret_adj.rolling(d.vol_window).std()
    out["roll_mean_ret"] = ret_adj.rolling(d.vol_window).mean()

    log_vol = np.log(vol.replace(0, np.nan))
    out["rel_volume"] = vol / vol.rolling(d.vol_window).mean()
    out["vol_change"] = log_vol.diff()
    out["amihud"] = ret_adj.abs() / (close * vol).replace(0, np.nan)

    for w in d.mom_windows:
        out[f"mom_{w}"] = np.log(adj / adj.shift(w))

    out["rsi"] = _rsi(close, d.rsi_window) / 100.0
    out["atr"] = _atr(df, d.atr_window)

    out["z_ret"] = (
        ret_adj - ret_adj.rolling(d.zscore_window).mean()
    ) / ret_adj.rolling(d.zscore_window).std()
    out["z_vol"] = (
        log_vol - log_vol.rolling(d.zscore_window).mean()
    ) / log_vol.rolling(d.zscore_window).std()
    out["ma_ratio"] = close / close.rolling(d.vol_window).mean() - 1.0

    # Assign by aligned index (not ``.values``, which strips the datetime tz).
    out.insert(0, "date", df["date"])
    out.insert(0, "asset", df["asset"])
    return out


def build_daily_features(daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a long frame ``[asset, date, *DAILY_FEATURES]``. NaNs from warmup /
    undefined ratios are left in place; preprocessing standardizes then zero-fills."""
    parts = [
        _per_asset(g.sort_values("date"), cfg)
        for _, g in daily.groupby("asset", sort=False)
    ]
    out = pd.concat(parts, ignore_index=True)
    return out[["asset", "date", *DAILY_FEATURES]]
