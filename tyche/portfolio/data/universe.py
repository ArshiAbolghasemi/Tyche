"""The tradable universe and ticker normalization.

The universe is the intersection of assets that have BOTH prices and news. Prices
(daily + intraday) exist for 5 symbols; news uses ``name`` with a ``GOOG`` spelling
for Alphabet, so we canonicalize to the price symbol (``GOOGL``). Everything
downstream keys on these canonical symbols, in this fixed order.
"""

from __future__ import annotations

import pandas as pd

# Canonical price symbols -> aliases seen in the news feed's ``name`` column.
_ALIASES: dict[str, set[str]] = {
    "AAPL": {"AAPL"},
    "AMZN": {"AMZN"},
    "GOOGL": {"GOOGL", "GOOG"},
    "MSFT": {"MSFT"},
    "NVDA": {"NVDA"},
}

UNIVERSE: list[str] = list(_ALIASES)

_ALIAS_TO_CANON: dict[str, str] = {
    alias: canon for canon, aliases in _ALIASES.items() for alias in aliases
}


def canonicalize(symbol: str) -> str | None:
    """Map a raw symbol/name to its canonical universe symbol, or ``None`` if the
    symbol is outside the universe. Handles exchange prefixes like ``NASDAQ:AAPL``
    and the doubled ``NASDAQ:NASDAQ:AAPL`` artifact by taking the last segment."""
    if not isinstance(symbol, str):
        return None
    tail = symbol.split(":")[-1].strip().upper()
    return _ALIAS_TO_CANON.get(tail)


def to_canonical(series: pd.Series) -> pd.Series:
    """Vectorized ``canonicalize`` over a Series (non-universe rows become NaN)."""
    return series.map(canonicalize)
