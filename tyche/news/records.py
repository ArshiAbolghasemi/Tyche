"""Schema layer — dataclass column registries, enums, and the output contract.

The pipeline passes pandas DataFrames between agents; each stage has a frozen
dataclass whose field names are the Python identifiers and whose field *values*
are the on-disk column strings. So ``Article.id == "article_id"`` and you can use
the field directly as a column key: ``df[Article.id]``. Grouping fields by stage
keeps a rename from drifting across six modules. ``OUTPUT_COLUMNS`` is the final
contract emitted per (article, ticker).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NeutralizationStatus(str, Enum):
    OK = "ok"
    NO_PRIOR_AVAILABLE = "no_prior_available"
    GROUP_TOO_SMALL = "group_too_small"


class SanityDirection(str, Enum):
    POS = "pos"
    NEG = "neg"
    NEU = "neu"


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


# --- Agent 2 — Summarizer output columns ---
@dataclass(frozen=True)
class Summary:
    text: str = "summary_text"  # bart-large-cnn abstractive summary of full_text
    n_tokens: str = "summary_n_tokens"  # bge-m3-tokenizer length of the summary


# --- Agent 3 — Embedder output columns ---
@dataclass(frozen=True)
class Embedding:
    n_tokens: str = "embedding_n_tokens"  # bge-m3-tokenizer length of the summary


# --- Agent 4 — Deduplicator output columns ---
# Near-duplicate summaries (reprints/syndications) are clustered per month; every row
# inherits its cluster representative's summary so the sentiment call runs once per
# cluster and the score is shared across the cluster's members.
@dataclass(frozen=True)
class Dedup:
    month: str = "dedup_month"  # calendar-month bucket the row was deduplicated within
    cluster_id: str = "dedup_cluster_id"  # month-scoped cluster label
    is_representative: str = "is_representative"  # True for the closest-to-centroid row
    representative_text: str = "representative_summary"  # summary that will be scored


# --- Agent 5 — Scorer output columns ---
@dataclass(frozen=True)
class Score:
    p_pos: str = "p_pos"
    p_neg: str = "p_neg"
    p_neu: str = "p_neu"
    span_score: str = "s"  # p_pos - p_neg
    model_revision: str = "model_revision"
    rationale: str = "sentiment_rationale"  # LLM's one-line justification


# --- Agent 5 — Scorer output columns (one score per deduplicated summary, ticker) ---
# Named ``Aggregate`` for continuity with the neutralizer/output contract; there is
# no span-aggregation step — the (deduplicated) summary is scored directly.
@dataclass(frozen=True)
class Aggregate:
    p_pos: str = "agg_p_pos"
    p_neg: str = "agg_p_neg"
    p_neu: str = "agg_p_neu"
    raw_score: str = "raw_score"


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
    Article.full_text,
    Aggregate.p_pos,
    Aggregate.p_neg,
    Aggregate.p_neu,
    Aggregate.raw_score,
    Neutralize.sentiment_final,
    Summary.text,
    Dedup.representative_text,
    Dedup.is_representative,
    Score.rationale,
    Neutralize.entity_prior_applied,
    Neutralize.shrinkage_weight_w,
    Score.model_revision,
    Neutralize.status,
]
