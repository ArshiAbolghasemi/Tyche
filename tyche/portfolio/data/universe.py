"""The tradable universe and ticker normalization.

The universe is resolved from the data rather than hard-coded: it is the set of
symbols carrying BOTH prices and news, capped at ``UniverseConfig.size`` and ranked
by liquidity. See ``resolve_universe`` for why the cap exists.

Symbols arrive in a few spellings — bare (``AAON``), exchange-qualified
(``NASDAQ:AAON``), and occasionally doubly-qualified (``NASDAQ:NASDAQ:AAON``) — so
everything is normalized to the bare, upper-cased form before joining.
"""

from __future__ import annotations

import pandas as pd

from tyche.common.logging import get_logger
from tyche.portfolio.config import UniverseConfig

log = get_logger(__name__)


def canonicalize(symbol: str) -> str | None:
    """Normalize a raw symbol to its bare, upper-cased form.

    Strips exchange prefixes by taking the last ``:``-separated segment, which also
    handles the doubled ``NASDAQ:NASDAQ:AAPL`` artifact seen in some feeds.
    """
    if not isinstance(symbol, str):
        return None
    tail = symbol.split(":")[-1].strip().upper()
    return tail or None


def to_canonical(series: pd.Series) -> pd.Series:
    """Vectorized ``canonicalize`` over a Series (unusable rows become NaN)."""
    return series.map(canonicalize)


def resolve_universe(
    daily: pd.DataFrame, news_assets: set[str], cfg: UniverseConfig
) -> list[str]:
    """Pick the traded cross-section from the available data.

    Selection, in order:

    1. Keep symbols that have news — a price-only name contributes nothing to the
       news branch and would train it on a constant.
    2. Optionally require a complete price series, so the covariance is estimated on
       a rectangular panel rather than one whose membership changes by day.
    3. Rank what survives by median dollar volume and take the top ``size``.

    The cap is a hardware constraint, not a modelling preference: the network emits
    an ``N x N`` covariance and factorizes it every forward pass, so cost grows as
    ``N^3``. ``cfg.symbols`` bypasses the ranking when you want a fixed list.
    """
    priced = set(daily["asset"].unique())
    candidates = priced & news_assets
    if not candidates:
        raise RuntimeError(
            f"no symbol has both prices ({len(priced)}) and news ({len(news_assets)}) — "
            "check that the two feeds use the same symbology"
        )

    frame = daily[daily["asset"].isin(candidates)]

    if cfg.require_full_history:
        counts = frame.groupby("asset")["date"].nunique()
        full = counts[counts == counts.max()].index
        dropped = len(candidates) - len(full)
        if dropped:
            log.info(
                "universe: dropped %d/%d symbols without the full %d-day history",
                dropped,
                len(candidates),
                int(counts.max()),
            )
        frame = frame[frame["asset"].isin(full)]

    liquidity = (frame["adj_close"] * frame["volume"]).groupby(frame["asset"]).median()
    if cfg.min_dollar_volume > 0:
        liquidity = liquidity[liquidity >= cfg.min_dollar_volume]

    if cfg.symbols:
        wanted = [canonicalize(s) for s in cfg.symbols]
        selected = [s for s in wanted if s in liquidity.index]
        missing = sorted(set(wanted) - set(selected))
        if missing:
            log.warning(
                "universe: %d requested symbol(s) unavailable and skipped: %s",
                len(missing),
                ", ".join(missing[:10]),
            )
    else:
        selected = liquidity.nlargest(max(1, cfg.size)).index.tolist()

    if not selected:
        raise RuntimeError("universe resolved to zero symbols after filtering")

    universe = sorted(selected)
    log.info(
        "universe: %d symbols from %d priced / %d with news (cap=%d) | %s%s",
        len(universe),
        len(priced),
        len(news_assets),
        cfg.size,
        ", ".join(universe[:10]),
        " ..." if len(universe) > 10 else "",
    )
    return universe
