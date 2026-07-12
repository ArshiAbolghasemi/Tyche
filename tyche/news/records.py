"""Schema layer — dataclass column registries, enums, and the output contract.

The pipeline passes pandas DataFrames between agents; each stage has a frozen
dataclass whose field names are the Python identifiers and whose field *values*
are the on-disk column strings. So ``Article.id == "article_id"`` and you can use
the field directly as a column key: ``df[Article.id]``. Grouping fields by stage
keeps a rename from drifting across six modules. ``OUTPUT_COLUMNS`` is the final
contract emitted per (article, ticker).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum


class NeutralizationStatus(str, Enum):
    OK = "ok"
    NO_PRIOR_AVAILABLE = "no_prior_available"
    GROUP_TOO_SMALL = "group_too_small"


class SanityDirection(str, Enum):
    POS = "pos"
    NEG = "neg"
    NEU = "neu"


def _columns(cls: type) -> list[str]:
    """Return the column-string values of a column dataclass, in declaration order."""
    return [str(f.default) for f in fields(cls)]


# --- Agent 1 — Ingest output columns ---
@dataclass(frozen=True)
class Article:
    id: str = "article_id"
    ticker: str = "ticker"
    name: str = "name"
    exchange: str = "exchange"
    type: str = "type"
    isin: str = "isin"
    group_key: str = "group_key"
    valid_time: str = "valid_time"
    transaction_time: str = "transaction_time"
    full_text: str = "full_text"


# --- Agent 2 — Segmentation output columns ---
@dataclass(frozen=True)
class Span:
    id: str = "span_id"
    text: str = "span_text"
    position_index: str = "position_index"
    n_tokens: str = "n_tokens"
    relevant: str = "relevant"


# --- Agent 3 — Scorer output columns ---
@dataclass(frozen=True)
class Score:
    p_pos: str = "p_pos"
    p_neg: str = "p_neg"
    p_neu: str = "p_neu"
    span_score: str = "s"  # p_pos - p_neg
    model_revision: str = "model_revision"


# --- Agent 4 — Aggregator output columns ---
@dataclass(frozen=True)
class Aggregate:
    p_pos: str = "agg_p_pos"
    p_neg: str = "agg_p_neg"
    p_neu: str = "agg_p_neu"
    raw_score: str = "raw_score"
    n_spans: str = "n_spans"
    n_relevant_spans: str = "n_relevant_spans"


# --- Agent 5 — Neutralizer output columns ---
@dataclass(frozen=True)
class Neutralize:
    entity_prior_applied: str = "entity_prior_applied"
    shrinkage_weight_w: str = "shrinkage_weight_w"
    sentiment_final: str = "sentiment_final"
    status: str = "neutralization_status"
    trading_day: str = (
        "trading_day"  # derived from valid_time, for the group×day z-score
    )


# Final output contract (order matters for the emitted table).
OUTPUT_COLUMNS: list[str] = [
    Article.id,
    Article.ticker,
    Article.name,
    Article.exchange,
    Article.isin,
    Article.group_key,
    Article.valid_time,
    Article.transaction_time,
    Aggregate.p_pos,
    Aggregate.p_neg,
    Aggregate.p_neu,
    Aggregate.raw_score,
    Neutralize.sentiment_final,
    Aggregate.n_spans,
    Aggregate.n_relevant_spans,
    Neutralize.entity_prior_applied,
    Neutralize.shrinkage_weight_w,
    Score.model_revision,
    Neutralize.status,
]
