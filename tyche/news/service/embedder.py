"""Agent 3 — Embedder. Summary text → dense ``BAAI/bge-m3`` embedding.

The embedder is a thin, weights-free wrapper over the hosted HuggingFace
``InferenceClient.feature_extraction`` (provider ``hf-inference``). It exists so the
Deduplicator can cluster near-duplicate summaries by cosine distance before the
(paid) LLM sentiment call, collapsing reprints/syndications to one representative.

bge-m3 has an 8192-token context window — larger than any summary the Summarizer
emits — so summaries are embedded whole; the cached tokenizer is used only for a
defensive length guard (mirroring the FinBERT token-guard pattern the Scorer used to
own) and is shared with the Summarizer's ``summary_n_tokens`` count.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from huggingface_hub import HfApi, InferenceClient
from transformers import AutoTokenizer

from tyche.common.config import settings
from tyche.common.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> InferenceClient:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set — the bge-m3 embedding InferenceClient needs a "
            "HuggingFace API token. Export it (export HF_TOKEN=...) or put it in "
            "tyche/common/.secrets.toml."
        )
    provider = str(settings.embedding.provider)
    client = InferenceClient(provider=provider, api_key=token)
    log.info(
        "embedder InferenceClient ready (provider=%s, model=%s)",
        provider,
        settings.embedding.name,
    )
    return client


@lru_cache(maxsize=1)
def get_tokenizer():
    """bge-m3 tokenizer (weights-free) — shared with the Summarizer's length guard."""
    name = str(settings.embedding.name)
    revision = str(settings.embedding.revision)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    log.info("loaded tokenizer for %s (token-guard only)", name)
    return tokenizer


@lru_cache(maxsize=1)
def get_embedding_revision() -> str:
    """Frozen embedding-model revision (commit hash) for reproducibility / logging."""
    name = str(settings.embedding.name)
    revision = str(settings.embedding.revision)
    try:
        info = HfApi().model_info(name, revision=revision)
        return info.sha or revision
    except Exception:  # pragma: no cover - offline / hub error
        return revision


def _truncate_to_limit(text: str) -> str:
    """Defensive guard: hard-cut a text to bge-m3's context window so the hosted
    endpoint never silently truncates. Summaries never approach 8192 tokens, so this
    effectively never fires — it only protects against pathological inputs."""
    max_tokens = int(settings.embedding.max_tokens)
    tokenizer = get_tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) <= max_tokens:
        return text
    kept = tokenizer.encode(text, add_special_tokens=False)[: max_tokens - 2]
    return tokenizer.decode(kept)


def _embed_one(text: str) -> np.ndarray:
    """Embed a single text into a 1-D float32 vector via the hosted endpoint."""
    client = _get_client()
    model = str(settings.embedding.name)
    raw = client.feature_extraction(_truncate_to_limit(text), model=model)
    vec = np.asarray(raw, dtype=np.float32)
    # feature_extraction may return (dim,) or (tokens, dim); mean-pool the latter.
    if vec.ndim > 1:
        vec = vec.mean(axis=tuple(range(vec.ndim - 1)))
    return vec.ravel()


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts concurrently, returning an ``(n, dim)`` float32 array.

    Order is preserved (results are written back by index), so the embedding of
    ``texts[i]`` is always row ``i`` regardless of completion order."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    max_workers = max(1, int(settings.embedding.max_workers))
    log.info(
        "embedding %d texts with %s (rev=%s, %d workers)",
        len(texts),
        settings.embedding.name,
        get_embedding_revision(),
        max_workers,
    )
    vectors: list[np.ndarray | None] = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, vec in zip(range(len(texts)), pool.map(_embed_one, texts)):
            vectors[i] = vec
    return np.vstack(vectors)
