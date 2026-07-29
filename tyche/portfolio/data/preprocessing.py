"""Standardization fitted on the training split only, reused unchanged elsewhere.

Per-feature mean/std are computed over training-day observations, then applied to
every split; remaining NaNs (warmup, undefined ratios, missing bars) are zero-filled
*after* standardization. Binary columns (the news ``no_news`` flag) are passed
through untouched. Intraday stats ignore zero-padding rows so padding does not skew
the moments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tyche.portfolio.data.assemble import AlignedData


@dataclass
class Standardizer:
    daily_mean: np.ndarray
    daily_std: np.ndarray
    news_mean: np.ndarray
    news_std: np.ndarray
    intra_mean: np.ndarray
    intra_std: np.ndarray
    news_passthrough: list[int]  # column indices left unscaled (binary flags)


def _moments(x: np.ndarray, axis: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_standardizer(data: AlignedData, train_day_idx: np.ndarray) -> Standardizer:
    d = data.daily[:, train_day_idx, :]
    n = data.news[:, train_day_idx, :]
    daily_mean, daily_std = _moments(d, axis=(0, 1))
    news_mean, news_std = _moments(n, axis=(0, 1))

    # Intraday: mask fully-zero (padding) bars out of the moment estimate.
    it = data.intraday[:, train_day_idx, :, :]
    flat = it.reshape(-1, it.shape[-1])
    nonpad = flat[np.abs(flat).sum(axis=1) > 0]
    intra_mean, intra_std = _moments(nonpad, axis=(0,))

    passthrough = (
        [data.news_names.index("no_news")] if "no_news" in data.news_names else []
    )
    return Standardizer(
        daily_mean,
        daily_std,
        news_mean,
        news_std,
        intra_mean,
        intra_std,
        passthrough,
    )


def apply_standardizer(data: AlignedData, s: Standardizer) -> AlignedData:
    daily = np.nan_to_num((data.daily - s.daily_mean) / s.daily_std)

    news = (data.news - s.news_mean) / s.news_std
    for c in s.news_passthrough:  # restore raw binary flag columns
        news[:, :, c] = data.news[:, :, c]
    news = np.nan_to_num(news)

    intra = np.nan_to_num((data.intraday - s.intra_mean) / s.intra_std)
    # Re-zero padding rows so standardization doesn't turn pads into -mean/std.
    pad = np.abs(data.intraday).sum(axis=-1, keepdims=True) == 0
    intra = np.where(pad, 0.0, intra).astype(np.float32)

    return AlignedData(
        assets=data.assets,
        days=data.days,
        daily=daily.astype(np.float32),
        news=news.astype(np.float32),
        intraday=intra,
        adj_close=data.adj_close,
        daily_names=data.daily_names,
        news_names=data.news_names,
        intraday_names=data.intraday_names,
    )
