"""Agent 2 — Summarizer. Article ``full_text`` → one abstractive summary per row.

Replaces the old sentence-Segmentation stage. Instead of exploding an article into
sentence spans, we compress the whole article to a short summary with
``facebook/bart-large-cnn`` and hand that single summary to the Scorer — so there is
one FinBERT score per (article, ticker) and no downstream aggregation.

Runs the model **locally** (``AutoModelForSeq2SeqLM`` + ``generate``) rather than the
hosted InferenceClient: the hosted endpoint does not truncate over-long inputs and
crashes on BART's 1024-token positional-embedding limit ("index out of range in
self"). Loading in-process lets us truncate each input at 1024 tokens explicitly and
batch. (transformers 5.x dropped the ``"summarization"`` pipeline task, so we call the
model directly.)

Design guards
-------------
* ``truncation=True`` caps each article at BART's input window — no positional-index
  overflow on long articles.
* ``max_length`` (config) is kept well under FinBERT's 512-token window so the summary
  is never silently truncated when scored; a FinBERT-tokenizer length guard in the
  Scorer is the belt-and-braces backstop.
* ``min_length`` protects against over-compression that would drop information.
* ``do_sample=False`` (beam search) → the same article always yields the same summary,
  preserving Audit C's causal (bit-for-bit) invariance. Identical ``full_text`` is
  summarized once and reused, so the multi-ticker fan-out costs one inference.
* Short articles (< ``min_words_to_summarize``) are already digestible — they are
  passed through verbatim, skipping inference.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from huggingface_hub import HfApi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from tyche.common.config import settings
from tyche.common.logging import get_logger
from tyche.news.agents.scorer import get_tokenizer
from tyche.news.records import Article, Summary

log = get_logger(__name__)


# BART's positional embeddings cap the input at 1024 tokens; truncate there so a long
# article never overflows ("index out of range in self").
_MAX_INPUT_TOKENS = 1024


def _resolve_device(device: str):
    """Map a device string to a torch device string ('cpu', 'cuda', 'mps', ...)."""
    d = (device or "cpu").strip().lower()
    if d in ("", "-1", "cpu"):
        return "cpu"
    if d.isdigit():
        return f"cuda:{d}"
    return d


@lru_cache(maxsize=1)
def _get_model():
    """Lazily load tokenizer + seq2seq weights (downloads on first use)."""

    cfg = settings.summarizer
    device = _resolve_device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.name), revision=str(cfg.revision))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(cfg.name), revision=str(cfg.revision)
    )
    model.to(device)
    model.eval()
    log.info(
        "summarization model ready (model=%s, rev=%s, device=%s)",
        cfg.name,
        cfg.revision,
        device,
    )
    return tokenizer, model, device


@lru_cache(maxsize=1)
def get_summarizer_revision() -> str:
    """Frozen summarizer revision (commit hash) for reproducibility / logging."""
    name = str(settings.summarizer.name)
    revision = str(settings.summarizer.revision)
    try:
        info = HfApi().model_info(name, revision=revision)
        return info.sha or revision
    except Exception:  # pragma: no cover - offline / hub error
        return revision


def _generate_kwargs() -> dict:
    cfg = settings.summarizer
    return {
        "min_length": int(cfg.min_length),
        "max_length": int(cfg.max_length),
        "num_beams": int(cfg.num_beams),
        "length_penalty": float(cfg.length_penalty),
        "do_sample": False,  # deterministic beam search — keeps Audit C bit-for-bit
        "early_stopping": True,
    }


def _summarize_batch(texts: list[str]) -> list[str]:
    """Summarize a list of articles locally via ``generate``, in one batched call.

    Inputs are truncated to BART's 1024-token window so long articles never overflow
    the positional embeddings."""
    if not texts:
        return []

    tokenizer, model, device = _get_model()
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_INPUT_TOKENS,
        padding=True,
    ).to(device)
    with torch.no_grad():
        ids = model.generate(**enc, **_generate_kwargs())
    return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in ids]


def summarize(ingested: pd.DataFrame) -> pd.DataFrame:
    """Add a ``summary_text`` column (one summary per row). Identical ``full_text``
    is summarized once and reused across the ticker fan-out."""
    if ingested.empty:
        out = ingested.copy()
        out[Summary.text] = pd.Series(dtype="string")
        out[Summary.n_tokens] = pd.Series(dtype="int")
        return out

    tokenizer = get_tokenizer()
    log.info(
        "summarizing %d rows with %s (rev=%s) params=%s",
        len(ingested),
        settings.summarizer.name,
        get_summarizer_revision(),
        _generate_kwargs(),
    )

    unique_texts = ingested[Article.full_text].dropna().unique().tolist()
    min_words = int(settings.summarizer.min_words_to_summarize)
    to_summarize = [t for t in unique_texts if len(t.split()) >= min_words]
    n_passthrough = len(unique_texts) - len(to_summarize)
    log.info(
        "%d unique articles across %d rows — %d to summarize, %d short passthroughs",
        len(unique_texts),
        len(ingested),
        len(to_summarize),
        n_passthrough,
    )

    cache: dict[str, str] = {t: t for t in unique_texts}  # short ones pass through
    batch = int(settings.summarizer.batch_size) * 25  # log cadence
    for start in range(0, len(to_summarize), batch):
        chunk = to_summarize[start : start + batch]
        for text, summary in zip(chunk, _summarize_batch(chunk)):
            cache[text] = summary or text  # never emit empty text to the scorer
        log.info(
            "summarized %d/%d articles",
            min(start + batch, len(to_summarize)),
            len(to_summarize),
        )

    out = ingested.copy()
    out[Summary.text] = out[Article.full_text].map(cache).fillna("")
    out[Summary.n_tokens] = [
        len(tokenizer.encode(s, add_special_tokens=True)) for s in out[Summary.text]
    ]
    over = int((out[Summary.n_tokens] > int(settings.model.max_tokens)).sum())
    log.info(
        "summarized %d rows (%d passthrough short articles); summary tokens "
        "min=%d mean=%.1f max=%d; %d over FinBERT's %d-token limit (scorer will guard)",
        len(out),
        n_passthrough,
        int(out[Summary.n_tokens].min()),
        float(out[Summary.n_tokens].mean()),
        int(out[Summary.n_tokens].max()),
        over,
        int(settings.model.max_tokens),
    )
    return out
