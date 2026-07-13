"""Agent 2 — Summarizer. Article ``full_text`` → one abstractive summary per row.

Replaces the old sentence-Segmentation stage. Instead of exploding an article into
sentence spans, we compress the whole article to a short summary with
``facebook/bart-large-cnn`` (hosted ``InferenceClient.summarization``) and hand that
single summary to the Scorer — so there is one FinBERT score per (article, ticker)
and no downstream aggregation.

Runs through the hosted InferenceClient (no local model weights). The endpoint does
NOT truncate over-long inputs itself and crashes on BART's 1024-token
positional-embedding limit ("index out of range in self"). ~35% of articles in this
corpus exceed that limit (some by 8x), so instead of truncating them down to the lead
paragraph, over-long articles are handled by **map-reduce summarization**: split into
sequential ≤1024-token chunks, summarize each chunk, then summarize the concatenation
of chunk-summaries down to the target length. This preserves information from the
whole article instead of only the first ~700 words.

Design guards
-------------
* Map-reduce chunking (not truncation) for inputs over the BART token limit — no
  positional-index overflow, and no information silently dropped from the tail of
  long articles.
* ``max_length`` (config) is kept well under FinBERT's 512-token window so the
  summary is never silently truncated when scored; a FinBERT-tokenizer length guard
  in the Scorer is the belt-and-braces backstop.
* ``min_length`` protects against over-compression that would drop information.
* Beam search with no sampling (``do_sample`` is not exposed by the endpoint, but
  omitting temperature/top_p keeps it deterministic) → the same article always yields
  the same summary, preserving Audit C's causal (bit-for-bit) invariance. Identical
  ``full_text`` is summarized once and reused, so the multi-ticker fan-out costs one
  round of API calls.
* Short articles (< ``min_words_to_summarize``) are already digestible — they are
  passed through verbatim, skipping the API call.
* Unique articles are summarized concurrently (thread pool; the work is I/O-bound
  API calls) — order-independent, so it doesn't affect Audit C's causal guarantee,
  which is about time ordering of the OUTPUT, not the order requests are issued in.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import pandas as pd
from huggingface_hub import HfApi, InferenceClient
from huggingface_hub.inference._generated.types import SummarizationOutput
from huggingface_hub.inference._providers import get_provider_helper
from transformers import AutoTokenizer

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
def _get_bart_tokenizer():
    """Weights-free BART tokenizer, used only to truncate inputs before they're sent
    to the hosted endpoint (mirrors the FinBERT token-guard pattern in the scorer)."""
    return AutoTokenizer.from_pretrained(
        str(settings.summarizer.name), revision=str(settings.summarizer.revision)
    )


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


def _chunk_to_bart_limit(text: str) -> list[str]:
    """Split ``text`` into sequential pieces that each fit BART's token window
    (config). A single-chunk list is returned unchanged when the text already fits.

    The safety margin is wider than just the BOS/EOS pair: BART's learned positional
    embeddings carry an internal offset of 2 beyond the special tokens, and a
    decode→re-encode round trip can occasionally add a token or two (whitespace/BPE
    boundary effects at chunk edges). ``-8`` covers both without meaningfully
    shrinking chunks."""
    max_tokens = int(settings.summarizer.max_tokens)
    tokenizer = _get_bart_tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False)
    budget = max_tokens - 8
    if len(ids) <= budget:
        return [text]
    return [tokenizer.decode(ids[i : i + budget]) for i in range(0, len(ids), budget)]


def _generate_parameters() -> dict:
    cfg = settings.summarizer
    return {
        "min_length": int(cfg.min_length),
        "max_length": int(cfg.max_length),
        "num_beams": int(cfg.num_beams),
        "length_penalty": float(cfg.length_penalty),
    }


def _call_summarization_api(text: str, parameters: dict) -> str:
    """One hosted-API summarization call. Assumes ``text`` already fits BART's token
    window (caller's responsibility — see ``_chunk_to_bart_limit``).

    ``InferenceClient.summarization`` nests the generation args under a
    ``generate_parameters`` key that the hf-inference (transformers) backend rejects
    ("model_kwargs not used"). We reproduce the client's own request path but pass the
    generation args FLAT under ``parameters`` — which is what the summarization
    pipeline actually expects.
    """
    client = _get_client()
    model = str(settings.summarizer.name)
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
    return (out.summary_text or "").strip()


def _summarize_one(text: str) -> str:
    """Summarize a single article. Short inputs pass through verbatim (no API call).

    Articles that fit BART's token window are summarized in one call. Longer articles
    are map-reduced: each ≤1024-token chunk is summarized independently (map), then
    the concatenation of chunk-summaries is summarized down to the target length
    (reduce) — preserving information from the whole article instead of truncating
    to the lead."""
    if len(text.split()) < int(settings.summarizer.min_words_to_summarize):
        return text

    chunks = _chunk_to_bart_limit(text)
    params = _generate_parameters()

    if len(chunks) == 1:
        summary = _call_summarization_api(chunks[0], params)
        return summary or text  # never emit empty text to the scorer

    chunk_summaries = [_call_summarization_api(c, params) for c in chunks]
    combined = " ".join(s for s in chunk_summaries if s)
    if not combined:
        return text
    # The reduce pass may itself exceed the token window for very long articles with
    # many chunks; recurse (chunk-summaries are much shorter than raw text, so this
    # converges quickly — typically one extra pass).
    reduced = _summarize_one(combined)
    return reduced or combined


def summarize(ingested: pd.DataFrame) -> pd.DataFrame:
    """Add a ``summary_text`` column (one summary per row). Identical ``full_text``
    is summarized once and reused across the ticker fan-out."""
    if ingested.empty:
        out = ingested.copy()
        out[Summary.text] = pd.Series(dtype="string")
        out[Summary.n_tokens] = pd.Series(dtype="int")
        return out

    tokenizer = get_tokenizer()
    max_workers = max(1, int(settings.summarizer.max_workers))
    log.info(
        "summarizing %d rows with %s (rev=%s) params=%s (chunk limit %d BART tokens, "
        "%d concurrent workers)",
        len(ingested),
        settings.summarizer.name,
        get_summarizer_revision(),
        _generate_parameters(),
        settings.summarizer.max_tokens,
        max_workers,
    )

    unique_texts = ingested[Article.full_text].dropna().unique().tolist()
    log.info(
        "%d unique articles across %d (article, ticker) rows — summarizing uniques "
        "concurrently (%d workers)",
        len(unique_texts),
        len(ingested),
        max_workers,
    )

    cache: dict[str, str] = {}
    n_passthrough = 0
    n_failed = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_text = {pool.submit(_summarize_one, t): t for t in unique_texts}
        for future in as_completed(future_to_text):
            text = future_to_text[future]
            try:
                summary = future.result()
            except Exception:
                log.exception(
                    "summarization failed for one article (len=%d chars) — "
                    "falling back to verbatim text",
                    len(text),
                )
                summary = text
                n_failed += 1
            if summary == text:
                n_passthrough += 1
            cache[text] = summary
            done += 1
            if done % 25 == 0 or done == len(unique_texts):
                log.info("summarized %d/%d unique articles", done, len(unique_texts))

    out = ingested.copy()
    out[Summary.text] = out[Article.full_text].map(cache).fillna("")
    out[Summary.n_tokens] = [
        len(tokenizer.encode(s, add_special_tokens=True)) for s in out[Summary.text]
    ]
    over = int((out[Summary.n_tokens] > int(settings.model.max_tokens)).sum())
    log.info(
        "summarized %d rows (%d passthrough short articles, %d API failures "
        "fell back to verbatim); summary tokens min=%d mean=%.1f max=%d; %d over "
        "FinBERT's %d-token limit (scorer will guard)",
        len(out),
        n_passthrough,
        n_failed,
        int(out[Summary.n_tokens].min()),
        float(out[Summary.n_tokens].mean()),
        int(out[Summary.n_tokens].max()),
        over,
        int(settings.model.max_tokens),
    )
    return out
