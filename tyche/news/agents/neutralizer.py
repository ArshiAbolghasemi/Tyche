"""Agent 5 — Neutralizer. Kills B2 (entity bias) in three strict steps.

0. Entity-prior correction: subtract the measured named-vs-masked offset per ticker
   (Audit B artifact), falling back to the group mean, else zero (flagged).
1. Causal robust rolling demean: subtract a STRICTLY TRAILING winsorized median of
   the ticker's own history (60d, the current row never enters its own correction),
   with hierarchical shrinkage toward the group for sparse names.
2. Group × day cross-sectional z-score: remove common-mode narrative, leaving
   idiosyncratic surprise on a comparable scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tyche.common.config import settings
from tyche.common.logging import get_logger
from tyche.news.records import (
    Aggregate,
    Article,
    NeutralizationStatus,
    Neutralize,
)

log = get_logger(__name__)


def load_entity_prior() -> dict:
    """Load the Audit-B artifact ``{by_ticker:{}, by_group:{}}`` if it exists."""
    path = Path(str(settings.neutralizer.entity_prior_path))
    if not path.exists():
        log.warning(
            "no entity_prior artifact at %s — corrections default to zero", path
        )
        return {"by_ticker": {}, "by_group": {}}
    return json.loads(path.read_text())


def _winsor_median(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    lo = np.quantile(arr, float(settings.neutralizer.winsor_lo))
    hi = np.quantile(arr, float(settings.neutralizer.winsor_hi))
    return float(np.median(np.clip(arr, lo, hi)))


def _trailing_location(
    df: pd.DataFrame, value_col: str, key: str
) -> tuple[pd.Series, pd.Series]:
    """Per-``key`` STRICTLY TRAILING winsorized-median location and obs count over a
    calendar window. For each row only rows with ``valid_time`` strictly earlier
    (``< t``, so same-timestamp rows are excluded too) within the window contribute
    — the causal guarantee. Returns Series aligned to ``df.index``."""
    window = np.timedelta64(int(settings.neutralizer.rolling_window_days), "D")
    loc = pd.Series(0.0, index=df.index)
    n = pd.Series(0.0, index=df.index)
    for _, sub in df.groupby(key, sort=False):
        sub = sub.sort_values(Article.valid_time)
        times = sub[Article.valid_time].to_numpy()
        vals = sub[value_col].to_numpy()
        idxs = sub.index.to_numpy()
        for j in range(len(sub)):
            mask = (times < times[j]) & (times >= times[j] - window)
            win = vals[mask]
            n.loc[idxs[j]] = len(win)
            loc.loc[idxs[j]] = _winsor_median(win)
    return loc, n


def neutralize(aggregated: pd.DataFrame) -> pd.DataFrame:
    df = aggregated.copy().reset_index(drop=True)
    df[Article.valid_time] = pd.to_datetime(df[Article.valid_time], utc=True)
    df[Neutralize.trading_day] = df[Article.valid_time].dt.date
    prior = load_entity_prior()
    by_ticker, by_group = prior.get("by_ticker", {}), prior.get("by_group", {})

    # --- Step 0: entity-prior correction ---
    status = pd.Series(NeutralizationStatus.OK.value, index=df.index)
    applied = np.zeros(len(df))
    for i, row in df.iterrows():
        if row[Article.ticker] in by_ticker:
            applied[i] = by_ticker[row[Article.ticker]]
        elif row[Article.group_key] in by_group:
            applied[i] = by_group[row[Article.group_key]]
        else:
            status[i] = NeutralizationStatus.NO_PRIOR_AVAILABLE.value
    df[Neutralize.entity_prior_applied] = applied
    df["_s0"] = df[Aggregate.raw_score].to_numpy() - applied

    # --- Step 1: causal robust rolling demean with hierarchical shrinkage ---
    loc_ticker, n_ticker = _trailing_location(df, "_s0", Article.ticker)
    loc_group, _ = _trailing_location(df, "_s0", Article.group_key)
    k = float(settings.neutralizer.shrinkage_k)
    w = n_ticker / (n_ticker + k)  # 0 when no history → fully borrow the group prior
    loc_shrunk = w * loc_ticker + (1.0 - w) * loc_group
    df[Neutralize.shrinkage_weight_w] = w
    df["_s1"] = df["_s0"] - loc_shrunk

    # --- Step 2: group × day cross-sectional z-score ---
    floor = float(settings.neutralizer.std_floor)
    min_members = int(settings.neutralizer.group_min_members)
    final = pd.Series(0.0, index=df.index)
    for _, idx in df.groupby(
        [Neutralize.trading_day, Article.group_key], sort=False
    ).groups.items():
        vals = df.loc[idx, "_s1"]
        if len(idx) < min_members:
            final.loc[idx] = vals
            status.loc[idx] = NeutralizationStatus.GROUP_TOO_SMALL.value
        else:
            final.loc[idx] = (vals - vals.mean()) / max(vals.std(ddof=0), floor)

    df[Neutralize.sentiment_final] = final
    df[Neutralize.status] = status
    df = df.drop(columns=["_s0", "_s1"])
    log.info(
        "neutralized %d rows (%d no_prior, %d group_too_small)",
        len(df),
        int((status == NeutralizationStatus.NO_PRIOR_AVAILABLE.value).sum()),
        int((status == NeutralizationStatus.GROUP_TOO_SMALL.value).sum()),
    )
    return df
