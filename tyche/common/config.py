"""Global configuration — a Dynaconf-derived class with env-var-backed properties.

There is no ``settings.toml``: every tunable is read from an environment variable
through a typed ``@property`` (with a built-in default). ``load_dotenv`` loads a
gitignored ``.env`` file automatically; copy ``.env.example`` to ``.env`` and edit.

The access shape mirrors the old nested config (``settings.model.name``,
``settings.neutralizer.rolling_window_days``, …) so agents don't change; each
section is a small ``@dataclass`` whose fields are env-sourced.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from dynaconf import Dynaconf
from dotenv import load_dotenv

# Load .env into os.environ once on import so the @property methods (which read
# os.environ directly) see values from the gitignored .env file.
load_dotenv()


def _env(key: str, default: Any, cast: type = str) -> Any:
    """Read ``key`` from the environment, casting to ``cast``; fall back to default."""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if cast is float:
        return float(raw)
    if cast is int:
        return int(raw)
    return raw


def _env_list(key: str, default: list[str]) -> list[str]:
    """Comma-separated env var → list[str]; JSON array if it starts with ``[``."""
    raw = os.environ.get(key)
    if not raw:
        return list(default)
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class PathsConfig:
    input: str = field(
        default_factory=lambda: _env("TYCHE_PATHS_INPUT", "data/eodhd/news.parquet")
    )
    output: str = field(
        default_factory=lambda: _env(
            "TYCHE_PATHS_OUTPUT", "data/output/news_sentiment.parquet"
        )
    )


@dataclass(frozen=True)
class IngestConfig:
    group_key_cols: list[str] = field(
        default_factory=lambda: _env_list(
            "TYCHE_INGEST_GROUP_KEY_COLS", ["exchange", "type"]
        )
    )
    masked_placeholder: str = field(
        default_factory=lambda: _env("TYCHE_INGEST_MASKED_PLACEHOLDER", "the company")
    )


@dataclass(frozen=True)
class SegmentationConfig:
    clause_delimiters: list[str] = field(
        default_factory=lambda: _env_list(
            "TYCHE_SEGMENTATION_CLAUSE_DELIMITERS",
            [";", ",", " and ", " but ", " because ", " although ", " however "],
        )
    )
    financial_vocab: list[str] = field(
        default_factory=lambda: _env_list(
            "TYCHE_SEGMENTATION_FINANCIAL_VOCAB",
            [
                "revenue",
                "revenues",
                "earnings",
                "profit",
                "loss",
                "income",
                "eps",
                "ebitda",
                "margin",
                "guidance",
                "outlook",
                "dividend",
                "acquisition",
                "merger",
                "deal",
                "growth",
                "decline",
                "surge",
                "drop",
                "beat",
                "miss",
                "forecast",
                "estimate",
                "quarter",
                "fiscal",
                "stock",
                "share",
                "shares",
                "market",
                "price",
                "target",
                "upgrade",
                "downgrade",
                "buy",
                "sell",
                "hold",
                "bullish",
                "bearish",
            ],
        )
    )


@dataclass(frozen=True)
class ModelConfig:
    name: str = field(
        default_factory=lambda: _env("TYCHE_MODEL_NAME", "ProsusAI/finbert")
    )
    revision: str = field(default_factory=lambda: _env("TYCHE_MODEL_REVISION", "main"))
    expected_labels: list[str] = field(
        default_factory=lambda: _env_list(
            "TYCHE_MODEL_EXPECTED_LABELS", ["positive", "negative", "neutral"]
        )
    )
    max_tokens: int = field(
        default_factory=lambda: _env("TYCHE_MODEL_MAX_TOKENS", 512, int)
    )
    batch_size: int = field(
        default_factory=lambda: _env("TYCHE_MODEL_BATCH_SIZE", 32, int)
    )
    device: str = field(default_factory=lambda: _env("TYCHE_MODEL_DEVICE", "cpu"))
    provider: str = field(
        default_factory=lambda: _env("TYCHE_MODEL_PROVIDER", "hf-inference")
    )


@dataclass(frozen=True)
class SummarizerConfig:
    """Agent 2 — abstractive summarizer (``facebook/bart-large-cnn`` via the hosted
    ``InferenceClient.summarization``). Downstream, the summary is embedded with
    ``BAAI/bge-m3`` (8192-token context) for dedup and read by the LLM sentiment
    scorer — neither of which imposes FinBERT's old 512-token cap — so ``max_length``
    can be larger, retaining more of the article. ``min_length`` still guards against
    over-compression that would drop information. Beam search (no sampling) keeps
    output deterministic — important for Audit C.
    """

    name: str = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_NAME", "facebook/bart-large-cnn")
    )
    revision: str = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_REVISION", "main")
    )
    provider: str = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_PROVIDER", "hf-inference")
    )
    min_length: int = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_MIN_LENGTH", 80, int)
    )
    # Bumped from 200 → 512: the summary now feeds bge-m3 (8192-token context) and an
    # LLM sentiment scorer, so it no longer has to fit FinBERT's 512-token window and
    # can retain more of the article. Still well under bart-large-cnn's 1024-token
    # generation limit.
    max_length: int = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_MAX_LENGTH", 512, int)
    )
    num_beams: int = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_NUM_BEAMS", 4, int)
    )
    length_penalty: float = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_LENGTH_PENALTY", 2.0, float)
    )
    # Below this many words the source is already short — score it verbatim and
    # skip the summarization API call entirely.
    min_words_to_summarize: int = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_MIN_WORDS", 80, int)
    )
    # BART's positional embeddings cap the input at 1024 tokens. Articles longer than
    # this are map-reduced (chunked, each chunk summarized, chunk-summaries then
    # summarized together) instead of truncated, so long articles aren't silently cut.
    max_tokens: int = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_MAX_TOKENS", 1024, int)
    )
    # Concurrent hosted-API requests during summarization (thread pool; I/O bound).
    max_workers: int = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_MAX_WORKERS", 8, int)
    )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Agent 3 — Embedder (``BAAI/bge-m3`` via the hosted ``InferenceClient``).

    Summaries are embedded into dense vectors so near-duplicate articles can be
    clustered and collapsed before the (paid) LLM sentiment call. bge-m3 has an
    8192-token context window — comfortably larger than any summary the summarizer
    emits — so summaries are embedded whole; ``max_tokens`` is only a defensive guard.
    """

    name: str = field(
        default_factory=lambda: _env("TYCHE_EMBEDDING_NAME", "BAAI/bge-m3")
    )
    revision: str = field(
        default_factory=lambda: _env("TYCHE_EMBEDDING_REVISION", "main")
    )
    provider: str = field(
        default_factory=lambda: _env("TYCHE_EMBEDDING_PROVIDER", "hf-inference")
    )
    # bge-m3's context window; summaries never approach it, so this is a safety cap.
    max_tokens: int = field(
        default_factory=lambda: _env("TYCHE_EMBEDDING_MAX_TOKENS", 8192, int)
    )
    # Concurrent hosted feature-extraction requests (thread pool; I/O bound).
    max_workers: int = field(
        default_factory=lambda: _env("TYCHE_EMBEDDING_MAX_WORKERS", 8, int)
    )


@dataclass(frozen=True)
class DedupConfig:
    """Agent 4 — Deduplicator. Collapses near-duplicate summaries so each cluster of
    reprints/syndications costs a single sentiment call.

    News is deduplicated one calendar month at a time (``window``): within a month,
    unique summaries are embedded, agglomeratively clustered by cosine distance, and
    each cluster is represented by the member closest to the cluster centroid. Every
    row inherits its cluster representative's summary, so downstream scoring is done
    once per cluster and the score is shared across the cluster's members.
    """

    # Cosine-distance threshold for agglomerative clustering: members within this
    # distance of each other are treated as the same story. Lower ⇒ stricter (fewer,
    # tighter clusters); ~0.10–0.20 captures reprints without merging distinct stories.
    distance_threshold: float = field(
        default_factory=lambda: _env("TYCHE_DEDUP_DISTANCE_THRESHOLD", 0.15, float)
    )
    # Pandas period-frequency alias for the dedup window; "M" = calendar month.
    window: str = field(default_factory=lambda: _env("TYCHE_DEDUP_WINDOW", "M"))


_DEFAULT_SENTIMENT_SYSTEM_PROMPT = """\
You are a senior financial-markets sentiment analyst. You read a short summary of a \
news item about a specific publicly-traded company or security and judge its likely \
sentiment IMPACT ON THAT SECURITY from the perspective of an investor holding it.

Return a probability distribution over exactly three classes:
- POSITIVE — the news is, on balance, favorable for the security (would tend to push \
its price up or reflects improving fundamentals). Examples: earnings/revenue beats, \
raised guidance, new large contracts, successful product launches, accretive M&A, \
buybacks, analyst upgrades, resolved litigation in the company's favor.
- NEGATIVE — the news is, on balance, unfavorable for the security (would tend to push \
its price down or reflects deteriorating fundamentals). Examples: earnings/revenue \
misses, cut or withdrawn guidance, profit warnings, regulatory penalties, lawsuits, \
recalls, executive departures under pressure, downgrades, dilution, dividend cuts.
- NEUTRAL — the news is factual/administrative with no clear directional implication, \
is purely informational (scheduling, routine disclosures), is mixed with offsetting \
positives and negatives, or does not actually concern the security's prospects.

Guidelines:
- Judge sentiment for the SECURITY/COMPANY, not the mood of the prose. "Shares fell on \
profit-taking despite a strong quarter" still describes a strong quarter — weigh the \
fundamental substance and the stated market reaction together.
- Distinguish the company's OWN prospects from broad market/sector commentary that only \
mentions it in passing; the latter leans NEUTRAL.
- Forward-looking guidance and analyst actions usually dominate backward-looking figures.
- Be calibrated: reserve high confidence (>0.8 in one class) for unambiguous news; when \
signals conflict or are weak, spread probability mass and lean NEUTRAL.
- The three probabilities must be non-negative and sum to 1.
- Provide a single concise sentence of rationale citing the key driver.

Respond ONLY via the structured schema you are given."""


@dataclass(frozen=True)
class SentimentConfig:
    """Agent 5 — Sentiment scorer (Azure OpenAI ``gpt-4.0-mini`` via LangChain).

    Replaces FinBERT: the deduplicated summary is sent to an Azure OpenAI chat model
    with a comprehensive financial-sentiment system prompt, and the model returns
    calibrated positive/negative/neutral probabilities (validated with pydantic,
    retried with tenacity). ``endpoint``/``deployment``/``api_version`` reconstruct
    the Azure REST URL; ``api_key`` must be supplied via env (never hardcoded).
    """

    endpoint: str = field(
        default_factory=lambda: _env(
            "TYCHE_SENTIMENT_ENDPOINT",
            "https://zanistagpteastus2.openai.azure.com",
        )
    )
    deployment: str = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_DEPLOYMENT", "gpt-4o-mini")
    )
    api_version: str = field(
        default_factory=lambda: _env(
            "TYCHE_SENTIMENT_API_VERSION", "2024-12-01-preview"
        )
    )
    api_key: str = field(default_factory=lambda: _env("TYCHE_SENTIMENT_API_KEY", ""))
    temperature: float = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_TEMPERATURE", 0.0, float)
    )
    # tenacity retry budget for transient Azure errors (rate limits, 5xx, timeouts).
    max_retries: int = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_MAX_RETRIES", 5, int)
    )
    request_timeout: float = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_TIMEOUT", 60.0, float)
    )
    # Concurrent sentiment calls (thread pool; I/O bound). One call per unique cluster
    # representative, so this is the effective sentiment-throughput knob.
    max_workers: int = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_MAX_WORKERS", 8, int)
    )
    # The financial-sentiment system prompt sent with every scoring call. Overridable
    # via env (e.g. to A/B a prompt variant) without a code change; defaults to the
    # comprehensive prompt above.
    system_prompt: str = field(
        default_factory=lambda: _env(
            "TYCHE_SENTIMENT_SYSTEM_PROMPT", _DEFAULT_SENTIMENT_SYSTEM_PROMPT
        )
    )


@dataclass(frozen=True)
class AggregationConfig:
    position_lambda: float = field(
        default_factory=lambda: _env("TYCHE_AGGREGATION_POSITION_LAMBDA", 1.5, float)
    )
    irrelevant_discount: float = field(
        default_factory=lambda: _env(
            "TYCHE_AGGREGATION_IRRELEVANT_DISCOUNT", 0.3, float
        )
    )
    weight_epsilon: float = field(
        default_factory=lambda: _env("TYCHE_AGGREGATION_WEIGHT_EPSILON", 1e-6, float)
    )


@dataclass(frozen=True)
class NeutralizerConfig:
    entity_prior_path: str = field(
        default_factory=lambda: _env(
            "TYCHE_NEUTRALIZER_ENTITY_PRIOR_PATH", "data/output/entity_prior.json"
        )
    )
    rolling_window_days: int = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_ROLLING_WINDOW_DAYS", 60, int)
    )
    min_events: int = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_MIN_EVENTS", 10, int)
    )
    shrinkage_k: float = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_SHRINKAGE_K", 20.0, float)
    )
    winsor_lo: float = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_WINSOR_LO", 0.01, float)
    )
    winsor_hi: float = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_WINSOR_HI", 0.99, float)
    )
    std_floor: float = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_STD_FLOOR", 1e-6, float)
    )
    group_min_members: int = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_GROUP_MIN_MEMBERS", 3, int)
    )


_DEFAULT_SANITY = [
    {"text": "revenues increased significantly", "expect": "pos"},
    {"text": "the company reported a loss", "expect": "neg"},
    {"text": "the meeting was held on Tuesday", "expect": "neu"},
    {"text": "profit beat expectations and the stock surged", "expect": "pos"},
    {"text": "shares tumbled after the profit warning", "expect": "neg"},
]


def _env_sanity_sentences() -> list[dict]:
    """Parse ``TYCHE_AUDITOR_SANITY_SENTENCES`` (JSON list of {text, expect}) else default."""
    raw = os.environ.get("TYCHE_AUDITOR_SANITY_SENTENCES")
    if not raw:
        return list(_DEFAULT_SANITY)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return list(_DEFAULT_SANITY)


@dataclass(frozen=True)
class AuditorConfig:
    baseline_path: str = field(
        default_factory=lambda: _env(
            "TYCHE_AUDITOR_BASELINE_PATH", "data/output/baseline.json"
        )
    )
    psi_threshold: float = field(
        default_factory=lambda: _env("TYCHE_AUDITOR_PSI_THRESHOLD", 0.10, float)
    )
    same_sign_alert: float = field(
        default_factory=lambda: _env("TYCHE_AUDITOR_SAME_SIGN_ALERT", 0.80, float)
    )
    sanity_sentences: list[dict] = field(default_factory=_env_sanity_sentences)


@dataclass(frozen=True)
class DaskConfig:
    blocksize: str = field(
        default_factory=lambda: _env("TYCHE_DASK_BLOCKSIZE", "128MB")
    )
    npartitions: int = field(
        default_factory=lambda: _env("TYCHE_DASK_NPARTITIONS", 4, int)
    )


class TycheSettings(Dynaconf):
    """Dynaconf subclass that exposes all tunables as env-var-backed ``@property``.

    No settings file is used — values come from environment variables (loaded from a
    gitignored ``.env`` via ``load_dotenv``). Nested access (``settings.model.name``)
    returns a frozen dataclass section built from the current environment, so the
    config always reflects the live env at access time. The ``TYCHE_ENV`` variable
    selects a deployment profile (development / staging / production).
    """

    def __init__(self, **kwargs):
        env = os.environ.get("TYCHE_ENV", "development").lower()
        merged = dict(
            settings_files=[],
            environments=True,
            env=env,
            envvar_prefix="TYCHE",
            load_dotenv=True,
        )
        merged.update(kwargs)
        super().__init__(**merged)

    @property
    def paths(self) -> PathsConfig:
        return PathsConfig()

    @property
    def ingest(self) -> IngestConfig:
        return IngestConfig()

    @property
    def segmentation(self) -> SegmentationConfig:
        return SegmentationConfig()

    @property
    def model(self) -> ModelConfig:
        return ModelConfig()

    @property
    def summarizer(self) -> SummarizerConfig:
        return SummarizerConfig()

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig()

    @property
    def dedup(self) -> DedupConfig:
        return DedupConfig()

    @property
    def sentiment(self) -> SentimentConfig:
        return SentimentConfig()

    @property
    def aggregation(self) -> AggregationConfig:
        return AggregationConfig()

    @property
    def neutralizer(self) -> NeutralizerConfig:
        return NeutralizerConfig()

    @property
    def auditor(self) -> AuditorConfig:
        return AuditorConfig()

    @property
    def dask(self) -> DaskConfig:
        return DaskConfig()


settings = TycheSettings()
