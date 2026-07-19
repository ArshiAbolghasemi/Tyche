"""Agent 3 — Embedder. Summary text → dense ``BAAI/bge-m3`` embedding.

Weights are loaded directly onto a local device (CPU/CUDA/MPS, see
``tyche.common.device``) — no hosted API call — so embedding throughput is bounded by
local hardware, not an external rate limit. It exists so the Deduplicator can cluster
near-duplicate summaries by cosine distance before the (paid) LLM sentiment call,
collapsing reprints/syndications to one representative.

The dense embedding follows bge-m3's documented pooling: the CLS token of the last
hidden state, L2-normalized — so cosine similarity between two embeddings is a plain
dot product, exactly what the Deduplicator's cosine-distance clustering assumes.

bge-m3 has an 8192-token context window — larger than any summary the Summarizer
emits — so summaries are embedded whole (truncation only guards pathological inputs).
The tokenizer is shared with the Summarizer's ``summary_n_tokens`` diagnostic.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from tyche.common.config import settings
from tyche.common.device import resolve_device
from tyche.common.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_tokenizer():
    """bge-m3 tokenizer — shared with the Summarizer's length guard."""
    name = str(settings.embedding.name)
    revision = str(settings.embedding.revision)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    log.info("loaded tokenizer for %s", name)
    return tokenizer


@lru_cache(maxsize=1)
def _get_device() -> torch.device:
    return resolve_device(str(settings.embedding.device))


@lru_cache(maxsize=1)
def _get_model():
    name = str(settings.embedding.name)
    revision = str(settings.embedding.revision)
    device = _get_device()
    model = AutoModel.from_pretrained(name, revision=revision)
    model.to(device)
    model.eval()
    log.info("loaded %s (rev=%s) onto device=%s", name, revision, device)
    return model


@lru_cache(maxsize=1)
def get_embedding_revision() -> str:
    """Frozen embedding-model revision (commit hash) for reproducibility / logging."""
    from huggingface_hub import HfApi

    name = str(settings.embedding.name)
    revision = str(settings.embedding.revision)
    try:
        info = HfApi().model_info(name, revision=revision)
        return info.sha or revision
    except Exception:  # pragma: no cover - offline / hub error
        return revision


def _embed_batch(texts: list[str]) -> np.ndarray:
    """Embed one batch of texts on the local device: CLS-pool the last hidden state
    and L2-normalize, per bge-m3's documented dense-embedding recipe."""
    tokenizer = get_tokenizer()
    model = _get_model()
    device = _get_device()
    max_tokens = int(settings.embedding.max_tokens)

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    ).to(device)
    with torch.inference_mode():
        output = model(**encoded)
    cls = output.last_hidden_state[:, 0]
    normalized = F.normalize(cls, p=2, dim=1)
    return normalized.cpu().numpy().astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts on the local device, batched for throughput. Order is
    preserved: row ``i`` of the result is the embedding of ``texts[i]``."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    batch_size = max(1, int(settings.embedding.batch_size))
    log.info(
        "embedding %d texts with %s (rev=%s) on device=%s (batch_size=%d)",
        len(texts),
        settings.embedding.name,
        get_embedding_revision(),
        _get_device(),
        batch_size,
    )
    batches = [
        _embed_batch(texts[i : i + batch_size])
        for i in range(0, len(texts), batch_size)
    ]
    return np.vstack(batches)
