"""Windowed cross-sectional samples with a strictly chronological, purged split.

A sample at decision day ``t`` carries the synchronized ``T``-day lookback across
all assets/branches and a target of forward ``H``-day log returns (t -> t+H). Splits
are assigned by decision date; the target window ``[t, t+H]`` must not cross a split
boundary, and an ``embargo`` (>= H) of trading days is dropped after each boundary so
no train sample's horizon overlaps validation/test. No shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tyche.portfolio.data.assemble import AlignedData
from tyche.portfolio.config import Config


@dataclass
class Sample:
    t: int  # decision-day index into AlignedData.days
    day_slice: slice  # lookback [t-T+1, t+1)
    target: np.ndarray  # [A] forward H-day log returns


@dataclass
class SplitIndex:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


def _forward_return(adj_close: np.ndarray, t: int, h: int) -> np.ndarray:
    return np.log(adj_close[:, t + h] / adj_close[:, t])


def _label_of(day: pd.Timestamp, cfg: Config) -> str | None:
    s = cfg.split
    if (
        pd.Timestamp(s.train_start, tz="UTC")
        <= day
        <= pd.Timestamp(s.train_end, tz="UTC")
    ):
        return "train"
    if pd.Timestamp(s.val_start, tz="UTC") <= day <= pd.Timestamp(s.val_end, tz="UTC"):
        return "val"
    if (
        pd.Timestamp(s.test_start, tz="UTC")
        <= day
        <= pd.Timestamp(s.test_end, tz="UTC")
    ):
        return "test"
    return None


def build_splits(data: AlignedData, cfg: Config) -> SplitIndex:
    T, H = cfg.window.lookback, cfg.window.holding
    embargo = max(cfg.window.embargo, H)
    days = data.days
    n = len(days)

    buckets: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    day_labels = [_label_of(d, cfg) for d in days]

    for t in range(T - 1, n - H):
        label = day_labels[t]
        if label is None:
            continue
        # Purge: the whole target horizon must stay inside this split.
        if any(day_labels[k] != label for k in range(t, t + H + 1)):
            continue
        sample = Sample(
            t=t,
            day_slice=slice(t - T + 1, t + 1),
            target=_forward_return(data.adj_close, t, H).astype(np.float32),
        )
        buckets[label].append(sample)

    return SplitIndex(**_apply_embargo(buckets, day_labels, embargo))


def _apply_embargo(buckets, day_labels, embargo):
    """Drop the first ``embargo`` samples of val/test (adjacent to the prior split)
    so no residual horizon overlap survives across a boundary."""

    def trim(samples: list[Sample]) -> list[Sample]:
        if not samples:
            return samples
        cutoff = samples[0].t + embargo
        return [s for s in samples if s.t >= cutoff]

    return {
        "train": buckets["train"],
        "val": trim(buckets["val"]),
        "test": trim(buckets["test"]),
    }


def train_day_indices(splits: SplitIndex, cfg: Config) -> np.ndarray:
    """Every trading-day index touched by a training sample's lookback — the only
    days preprocessing is allowed to see."""
    idx: set[int] = set()
    for s in splits.train:
        idx.update(range(s.day_slice.start, s.day_slice.stop))
    return np.array(sorted(idx), dtype=int)
