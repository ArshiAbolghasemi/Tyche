"""Agent 2 — Summarizer. Article ``full_text`` → one abstractive summary per row.

Replaces the old sentence-Segmentation stage. Instead of exploding an article into
sentence spans, we compress the whole article to a short summary with
``facebook/bart-large-cnn`` (hosted ``InferenceClient.summarization``) and hand that
single summary to the Scorer — so there is one FinBERT score per (article, ticker)
and no downstream aggregation.

Design guards
-------------
* ``max_length`` (config) is kept well under FinBERT's 512-token window so the
  summary is never silently truncated when scored; a FinBERT-tokenizer length guard
  in the Scorer is the belt-and-braces backstop.
* ``min_length`` protects against over-compression that would drop information.
* Beam search with no sampling → the same article always yields the same summary,
  preserving Audit C's causal (bit-for-bit) invariance. Identical ``full_text`` is
  summarized once and reused, so the multi-ticker fan-out costs one API call.
* Short articles (< ``min_words_to_summarize``) are already digestible — they are
  passed through verbatim, skipping the API call.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from huggingface_hub import HfApi, InferenceClient
from huggingface_hub.inference._generated.types import SummarizationOutput
from huggingface_hub.inference._providers import get_provider_helper

from tyche.common.config import settings
from tyche.common.logging import get_logger
from tyche.news.agents.scorer import get_tokenizer
from tyche.news.records import Article, Summary

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> InferenceClient:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set — the summarization InferenceClient needs a "
            "HuggingFace API token. Export it (export HF_TOKEN=...) or put it in "
            "tyche/common/.secrets.toml."
        )
    provider = str(settings.summarizer.provider)
    client = InferenceClient(provider=provider, api_key=token)
    log.info(
        "summarizer InferenceClient ready (provider=%s, model=%s)",
        provider,
        settings.summarizer.name,
    )
    return client


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


def _generate_parameters() -> dict:
    cfg = settings.summarizer
    return {
        "min_length": int(cfg.min_length),
        "max_length": int(cfg.max_length),
        "num_beams": int(cfg.num_beams),
        "length_penalty": float(cfg.length_penalty),
    }


def _summarize_one(text: str) -> str:
    """Summarize a single article via the hosted endpoint. Short inputs pass through
    verbatim (no API call).

    ``InferenceClient.summarization`` nests the generation args under a
    ``generate_parameters`` key that the hf-inference (transformers) backend rejects
    ("model_kwargs not used"). We reproduce the client's own request path but pass the
    generation args FLAT under ``parameters`` — which is what the summarization
    pipeline actually expects — plus ``truncation`` to cap the BART input side.
    """
    if len(text.split()) < int(settings.summarizer.min_words_to_summarize):
        return text
    client = _get_client()
    model = str(settings.summarizer.name)
    parameters = {**_generate_parameters(), "truncation": "longest_first"}
    helper = get_provider_helper(client.provider, task="summarization", model=model)
    request = helper.prepare_request(
        inputs=text,
        parameters=parameters,
        headers=client.headers,
        model=model,
        api_key=client.token,
    )
    response = client._inner_post(request)
    out = SummarizationOutput.parse_obj_as_list(response)[0]
    summary = (out.summary_text or "").strip()
    return summary or text  # never emit empty text to the scorer


def summarize(ingested: pd.DataFrame) -> pd.DataFrame:
    """Add a ``summary_text`` column (one summary per row). Identical ``full_text``
    is summarized once and reused across the ticker fan-out."""
    if ingested.empty:
        out = ingested.copy()
        out[Summary.text] = pd.Series(dtype="string")
        out[Summary.n_tokens] = pd.Series(dtype="int")
        return out

    tokenizer = get_tokenizer()
    params = _generate_parameters()
    log.info(
        "summarizing %d rows with %s (rev=%s) params=%s",
        len(ingested),
        settings.summarizer.name,
        get_summarizer_revision(),
        params,
    )

    unique_texts = ingested[Article.full_text].dropna().unique().tolist()
    log.info(
        "%d unique articles across %d (article, ticker) rows — summarizing uniques",
        len(unique_texts),
        len(ingested),
    )

    cache: dict[str, str] = {}
    n_passthrough = 0
    for i, text in enumerate(unique_texts, 1):
        summary = _summarize_one(text)
        if summary is text or summary == text:
            n_passthrough += 1
        cache[text] = summary
        if i % 25 == 0:
            log.info("summarized %d/%d unique articles", i, len(unique_texts))

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
