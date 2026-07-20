"""Agent 1 — Ingest. Raw news file → clean (article, ticker) table.

Reads the source with **dask** (lazy, partitioned) instead of pandas, so the
multi-GB feed never needs to be fully materialized for the read itself. The source
is read into a dask DataFrame then computed to pandas for the row-level explode
(necessary because the ``symbols`` column may be JSON / multi-value and the
FinBERT scorer iterates per span via the hosted API). ``nrows`` bounds the source
read for dev / smoke runs.

Parses publication time (``valid_time``), stamps processing time
(``transaction_time``), builds the text field, explodes the multi-ticker
``symbols`` column into one row per (article, ticker), and assigns a stable
``group_key`` and per-article ``article_id`` (uuid, shared across a symbol set).
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

import dask.dataframe as dd
import pandas as pd

from tyche.common.config import settings
from tyche.common.logging import get_logger
from tyche.news.records import Article, Summary

log = get_logger(__name__)

_SOURCE_COLUMNS = [
    "ticker",
    "name",
    "Name",  # zanista source capitalizes the company-name column
    "exchange",
    "type",
    "isin",
    "date",
    "title",
    "content",
    "symbols",
    "summary",  # zanista source ships a pre-computed summary for ~35% of rows
]


def _read_dask(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Read CSV or parquet via dask, returning a computed pandas frame.

    Parquet is the recommended, efficiently-partitioned path. CSVs with
    embedded-newline quoted fields (news ``content``) cannot be safely
    block-partitioned — dask splits at byte boundaries and breaks quoted records
    that span a block — so the CSV path uses ``blocksize=None`` (single partition)
    to keep records intact. ``nrows`` caps the number of source rows materialized
    (truncated in pandas after the dask compute; for large bounded reads prefer
    parquet).
    """
    if path.endswith(".parquet"):
        ddf = dd.read_parquet(path, engine="pyarrow")
    else:
        ddf = dd.read_csv(path, dtype="string", blocksize=None)

    # Keep only the columns we care about (some sources carry extra ones).
    cols = [c for c in _SOURCE_COLUMNS if c in ddf.columns]
    ddf = ddf[cols]

    frame = ddf.compute()
    if nrows is not None:
        frame = frame.head(nrows)
    return frame


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return ""
    return str(value).strip()


def _build_text(title: str, content: str) -> str:
    if title and content:
        return f"{title}. {content}"
    return title or content


def _parse_symbols(raw: object, primary: str) -> list[str]:
    """Return the deduplicated ticker set for a row (primary first)."""
    text = _clean(raw)
    symbols: list[str] = []
    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                symbols = [str(s).strip() for s in parsed if str(s).strip()]
        except (json.JSONDecodeError, TypeError):
            symbols = [s.strip() for s in re.split(r"[;,]", text) if s.strip()]
    ordered = [primary] + symbols if primary else symbols
    seen: set[str] = set()
    return [s for s in ordered if s and not (s in seen or seen.add(s))]


def ingest(input_path: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    frame = _read_dask(input_path or settings.paths.input, nrows=nrows)
    now = datetime.now(timezone.utc)
    group_cols = list(settings.ingest.group_key_cols)

    rows: list[dict] = []
    for _, row in frame.iterrows():
        title, content = _clean(row.get("title")), _clean(row.get("content"))
        text = _build_text(title, content)
        if not text:
            continue  # drop rows with no title AND no content

        valid_time = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
        if pd.isna(valid_time):
            continue
        exchange, itype = _clean(row.get("exchange")), _clean(row.get("type"))
        name = _clean(row.get("name")) or _clean(row.get("Name"))
        group_key = ":".join(_clean(row.get(c)) or "NA" for c in group_cols)
        article_id = str(uuid.uuid1())  # one id per article, shared across its tickers
        # Carried through as-is; the summarizer skips generation for rows where this
        # is already non-empty and only summarizes the ones that need it.
        existing_summary = _clean(row.get("summary"))

        for ticker in _parse_symbols(row.get("symbols"), _clean(row.get("ticker"))):
            rows.append(
                {
                    Article.id: article_id,
                    Article.ticker: ticker,
                    Article.name: name,
                    Article.exchange: exchange,
                    Article.type: itype,
                    Article.isin: _clean(row.get("isin")),
                    Article.group_key: group_key,
                    Article.valid_time: valid_time.to_pydatetime(),
                    Article.transaction_time: now,
                    Article.full_text: text,
                    Summary.text: existing_summary,
                }
            )

    out = pd.DataFrame(rows)
    log.info(
        "ingested %d (article, ticker) rows from %s",
        len(out),
        input_path or settings.paths.input,
    )
    return out
