"""Agent 5 — Scorer. Deduplicated summary → financial sentiment probabilities.

This is the only agent that calls the sentiment model. Sentiment is extracted by an
**Azure OpenAI** chat model (``gpt-4o-mini``) reached through **LangChain**
(``AzureChatOpenAI``), not FinBERT. The model is given a comprehensive
financial-sentiment system prompt and asked to return calibrated
positive/negative/neutral probabilities for the described security, which are:

* validated against a **pydantic** schema (``SentimentScores`` — non-negative,
  renormalized to sum to 1), and
* retried on transient failures with **tenacity** (exponential backoff).

Only the *unique cluster representatives* produced by the Deduplicator are scored;
every row inherits its representative's score, so near-duplicate reprints cost a
single API call. Outputs mirror the old FinBERT contract — ``agg_p_pos/agg_p_neg/
agg_p_neu`` and ``raw_score = p_pos - p_neg`` in ``[-1, 1]`` — so the Neutralizer and
output schema are unchanged.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from tyche.common.config import settings
from tyche.common.logging import get_logger
from tyche.news.records import Aggregate, Dedup, Score, Summary

log = get_logger(__name__)


# --- Response schema (pydantic) ----------------------------------------------
class SentimentScores(BaseModel):
    """Validated LLM sentiment response.

    Probabilities are coerced to be non-negative and renormalized to sum to 1, so a
    model that returns slightly off-sum values (or a lone class) still yields a proper
    distribution. ``rationale`` is a short human-readable justification for auditing.
    """

    p_positive: float = Field(
        description="Probability the news is POSITIVE for the security's investors.",
        ge=0.0,
        le=1.0,
    )
    p_negative: float = Field(
        description="Probability the news is NEGATIVE for the security's investors.",
        ge=0.0,
        le=1.0,
    )
    p_neutral: float = Field(
        description="Probability the news is NEUTRAL / has no clear directional impact.",
        ge=0.0,
        le=1.0,
    )
    rationale: str = Field(
        default="",
        description="One concise sentence justifying the sentiment assessment.",
    )

    @field_validator("p_positive", "p_negative", "p_neutral")
    @classmethod
    def _clip_non_negative(cls, v: float) -> float:
        return max(0.0, float(v))

    @model_validator(mode="after")
    def _renormalize(self) -> "SentimentScores":
        total = self.p_positive + self.p_negative + self.p_neutral
        if total <= 0:
            # Degenerate response — fall back to fully neutral.
            self.p_positive, self.p_negative, self.p_neutral = 0.0, 0.0, 1.0
        else:
            self.p_positive /= total
            self.p_negative /= total
            self.p_neutral /= total
        return self

    def as_triplet(self) -> tuple[float, float, float]:
        """Return ``(p_pos, p_neg, p_neu)`` in the scorer's canonical order."""
        return self.p_positive, self.p_negative, self.p_neutral


# --- LLM client ---------------------------------------------------------------
# The system prompt sent with every scoring call comes from
# ``settings.sentiment.system_prompt`` (env-overridable via
# ``TYCHE_SENTIMENT_SYSTEM_PROMPT``; see ``tyche.common.config`` for the default).
@lru_cache(maxsize=1)
def _get_structured_llm():
    """Build the singleton Azure OpenAI chat model wrapped for structured output.

    ``TYCHE_SENTIMENT_API_KEY`` must be present in the environment (or ``.env``); the
    endpoint, deployment and API version come from settings.
    """
    from langchain_openai import AzureChatOpenAI

    cfg = settings.sentiment
    if not cfg.api_key:
        raise RuntimeError(
            "TYCHE_SENTIMENT_API_KEY is not set — the Azure OpenAI sentiment scorer "
            "needs an API key. Export it (export TYCHE_SENTIMENT_API_KEY=...) or put "
            "it in the gitignored .env file."
        )
    llm = AzureChatOpenAI(
        azure_endpoint=str(cfg.endpoint),
        azure_deployment=str(cfg.deployment),
        api_version=str(cfg.api_version),
        api_key=str(cfg.api_key),
        temperature=float(cfg.temperature),
        timeout=float(cfg.request_timeout),
        max_retries=0,  # retries are handled by tenacity around the call
    )
    log.info(
        "sentiment LLM ready (Azure deployment=%s, api-version=%s)",
        cfg.deployment,
        cfg.api_version,
    )
    return llm.with_structured_output(SentimentScores)


@lru_cache(maxsize=1)
def get_model_revision() -> str:
    """Model identity recorded on every output row (Azure deployment + api-version)."""
    cfg = settings.sentiment
    return f"azure:{cfg.deployment}@{cfg.api_version}"


def _score_call(text: str) -> SentimentScores:
    """One structured LLM call, retried on transient errors by the tenacity wrapper."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_structured_llm()
    result = llm.invoke(
        [
            SystemMessage(content=str(settings.sentiment.system_prompt)),
            HumanMessage(content=f"News summary:\n\n{text}"),
        ]
    )
    # ``with_structured_output`` already returns a validated ``SentimentScores``; the
    # explicit reconstruction re-runs the pydantic validators as a belt-and-braces
    # guard against provider quirks (e.g. a raw dict slipping through).
    return SentimentScores.model_validate(
        result if isinstance(result, dict) else result.model_dump()
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(int(settings.sentiment.max_retries)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(Exception),
)
def _score_one(text: str) -> tuple[float, float, float, str]:
    """Score a single summary, returning ``(p_pos, p_neg, p_neu, rationale)``.

    Empty text short-circuits to fully neutral (no API call)."""
    if not text or not text.strip():
        return 0.0, 0.0, 1.0, "empty summary"
    scores = _score_call(text)
    p_pos, p_neg, p_neu = scores.as_triplet()
    return p_pos, p_neg, p_neu, scores.rationale


def _score_unique(texts: list[str]) -> dict[str, tuple[float, float, float, str]]:
    """Score each unique text once, concurrently. Returns a text → triplet+rationale map."""
    max_workers = max(1, int(settings.sentiment.max_workers))
    log.info(
        "scoring %d unique summaries via Azure OpenAI (%d concurrent workers)",
        len(texts),
        max_workers,
    )
    cache: dict[str, tuple[float, float, float, str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = tqdm(
            pool.map(_score_one, texts), total=len(texts), desc="scorer", unit="summary"
        )
        for text, res in zip(texts, results):
            cache[text] = res
    log.info("scored %d unique summaries via Azure OpenAI", len(texts))
    return cache


def score(deduplicated: pd.DataFrame) -> pd.DataFrame:
    """Score each row's cluster-representative summary through Azure OpenAI — one call
    per unique representative, shared across the cluster's members. Emits per-row
    ``agg_p_pos/agg_p_neg/agg_p_neu`` and ``raw_score = p_pos - p_neg`` (in [-1, 1],
    ~0 when neutral dominates), carrying every upstream column through."""
    revision = get_model_revision()

    # Score the deduplicated representative when present; otherwise the raw summary.
    text_col = (
        Dedup.representative_text
        if Dedup.representative_text in deduplicated.columns
        else Summary.text
    )
    texts = deduplicated[text_col].fillna("").tolist()
    unique_texts = list(dict.fromkeys(texts))
    log.info(
        "scoring %d rows via Azure OpenAI (%s) — %d unique summaries to score",
        len(texts),
        revision,
        len(unique_texts),
    )
    cache = _score_unique(unique_texts)

    triplets = np.array([cache[t][:3] for t in texts], dtype=float).reshape(-1, 3)
    out = deduplicated.copy()
    out[Aggregate.p_pos] = triplets[:, 0]
    out[Aggregate.p_neg] = triplets[:, 1]
    out[Aggregate.p_neu] = triplets[:, 2]
    out[Aggregate.raw_score] = out[Aggregate.p_pos] - out[Aggregate.p_neg]
    out[Score.rationale] = [cache[t][3] for t in texts]
    out[Score.model_revision] = revision
    log.info(
        "scored %d rows via Azure OpenAI (raw_score mean=%.4f std=%.4f)",
        len(out),
        float(out[Aggregate.raw_score].mean()) if len(out) else 0.0,
        float(out[Aggregate.raw_score].std(ddof=0)) if len(out) else 0.0,
    )
    return out


def score_texts(texts: list[str]) -> np.ndarray:
    """Convenience: score a raw list of strings, returning an (n, 3) prob array in
    (p_pos, p_neg, p_neu) order. Used by Audit A sanity checks."""
    cache = _score_unique(list(dict.fromkeys(texts)))
    return np.array([cache[t][:3] for t in texts], dtype=float).reshape(-1, 3)
