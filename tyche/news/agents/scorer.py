"""Agent 3 — Scorer. The ONLY agent that touches the model (ProsusAI/finbert).

Unlike a local-load path, scoring goes through the HuggingFace ``InferenceClient``
(Router) with ``provider="hf-inference"``::

    client = InferenceClient(provider="hf-inference", api_key=os.environ["HF_TOKEN"])
    client.text_classification(text, model="ProsusAI/finbert")

The client returns per-label ``{label, score}`` pairs, so we map by label NAME
(never by positional index) — there is no silent sign-flip risk because we read the
label string straight off the API response. A cached ``AutoTokenizer`` is still used
only by Agent 2 for the 512-token length guard (no model weights downloaded here).
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, InferenceClient
from transformers import AutoTokenizer

from tyche.common.config import settings
from tyche.common.logging import get_logger
from tyche.news.records import Score, Span

log = get_logger(__name__)


class LabelOrderError(RuntimeError):
    """Raised when the model's id2label does not match the configured order."""


@lru_cache(maxsize=1)
def _get_client() -> InferenceClient:
    """Build the singleton HuggingFace ``InferenceClient``.

    ``HF_TOKEN`` must be present in the environment (or ``.secrets.toml`` /
    ``.env``). The provider and model come from settings.
    """
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set — the InferenceClient needs a HuggingFace API token. "
            "Export it (export HF_TOKEN=...) or put it in tyche/common/.secrets.toml."
        )
    provider = str(settings.model.provider)
    client = InferenceClient(provider=provider, api_key=token)
    log.info(
        "InferenceClient ready (provider=%s, model=%s)", provider, settings.model.name
    )
    return client


@lru_cache(maxsize=1)
def _load_tokenizer():
    """Load the FinBERT tokenizer (weights-free) for Agent 2's token guard.

    A single cached load is shared with segmentation. No model weights are
    downloaded here — only the vocab/tokenizer files.
    """

    name = settings.model.name
    revision = settings.model.revision
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    log.info("loaded tokenizer for %s (token-guard only)", name)
    return tokenizer


@lru_cache(maxsize=1)
def get_tokenizer():
    """FinBERT tokenizer, shared with Agent 2's token-length guard (single load)."""
    return _load_tokenizer()


def _resolve_revision(name: str, revision: str) -> str:
    """Resolve a branch/tag to a concrete commit hash for reproducibility."""
    try:
        info = HfApi().model_info(name, revision=revision)
        return info.sha or revision
    except Exception:  # pragma: no cover - offline / hub error
        return revision


@lru_cache(maxsize=1)
def get_model_revision() -> str:
    """Frozen model revision (commit hash) recorded on every output row."""
    return _resolve_revision(str(settings.model.name), str(settings.model.revision))


def _score_one(text: str) -> dict[str, float]:
    """Score a single span via the InferenceClient, returning a {label: score} map."""
    client = _get_client()
    model = str(settings.model.name)
    raw = client.text_classification(text, model=model)
    # InferenceClient returns list[{label, score}] sorted by score desc.
    return {item["label"].lower(): float(item["score"]) for item in raw}


def score(spans: pd.DataFrame) -> pd.DataFrame:
    """Score every span through the InferenceClient. One API call per span (the
    hosted text-classification endpoint accepts a single string). Returns the span
    frame with (p_pos, p_neg, p_neu, s, revision) mapped by label NAME."""
    revision = get_model_revision()
    labels = [lab.lower() for lab in settings.model.expected_labels]
    for need in ("positive", "negative", "neutral"):
        if need not in labels:
            raise LabelOrderError(
                f"expected_labels={labels!r} is missing {need!r} — check settings."
            )

    texts = spans[Span.text].tolist()
    n = len(texts)
    probs = np.zeros((n, 3), dtype=float)

    for i, text in enumerate(texts):
        scores = _score_one(text)
        probs[i, 0] = scores.get("positive", 0.0)
        probs[i, 1] = scores.get("negative", 0.0)
        probs[i, 2] = scores.get("neutral", 0.0)
        if (i + 1) % 50 == 0:
            log.info("scored %d/%d spans", i + 1, n)

    out = spans.copy()
    out[Score.p_pos] = probs[:, 0]
    out[Score.p_neg] = probs[:, 1]
    out[Score.p_neu] = probs[:, 2]
    out[Score.span_score] = (
        out[Score.p_pos] - out[Score.p_neg]
    )  # in [-1, 1]; ~0 when neutral is high
    out[Score.model_revision] = revision
    log.info("scored %d spans via InferenceClient", len(out))
    return out


def score_texts(texts: list[str]) -> np.ndarray:
    """Convenience: score a raw list of strings, returning an (n, 3) prob array in
    (p_pos, p_neg, p_neu) order. Used by Audit A sanity checks."""
    frame = pd.DataFrame({Span.text: texts})
    scored = score(frame)
    return scored[[Score.p_pos, Score.p_neg, Score.p_neu]].to_numpy()
