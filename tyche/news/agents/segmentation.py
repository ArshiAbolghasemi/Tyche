"""Agent 2 — Segmentation. Article text → sentence-level spans.

FinBERT was fine-tuned on Financial PhraseBank (single financial sentences), so it
scores short self-contained statements best. We split into sentences, guard the
512-token limit (BERT silently truncates otherwise → wrong scores), and flag each
span's rule-based relevance. Most extraction quality comes from this step.
"""

from __future__ import annotations

import re

import nltk
import pandas as pd

from tyche.common.config import settings
from tyche.common.logging import get_logger
from tyche.news.agents.scorer import get_tokenizer
from tyche.news.records import Article, Span

log = get_logger(__name__)

_NLTK_READY = False


def _sentence_split(text: str) -> list[str]:
    """nltk punkt sentence tokenizer with a lazy download; regex fallback offline."""
    global _NLTK_READY
    try:
        if not _NLTK_READY:
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
            _NLTK_READY = True
        return [s for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception:  # pragma: no cover - network/nltk unavailable
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def _split_on_clauses(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Recursively split an over-long span at clause boundaries until every piece
    is ≤ max_tokens. Falls back to a hard token-window cut if no boundary helps."""
    if _n_tokens(tokenizer, text) <= max_tokens:
        return [text]
    for delim in settings.segmentation.clause_delimiters:
        if delim in text:
            parts = [p.strip() for p in text.split(delim) if p.strip()]
            if len(parts) > 1:
                out: list[str] = []
                for part in parts:
                    out.extend(_split_on_clauses(part, tokenizer, max_tokens))
                return out
    # No usable boundary: hard-cut by tokens so nothing silently truncates.
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = [ids[i : i + max_tokens - 2] for i in range(0, len(ids), max_tokens - 2)]
    return [tokenizer.decode(c) for c in chunks]


def _is_relevant(text: str, ticker: str, name: str) -> bool:
    low = text.lower()
    if any(term in low for term in settings.segmentation.financial_vocab):
        return True
    if ticker and ticker.lower() in low:
        return True
    return bool(name) and name.lower() in low


def segment(ingested: pd.DataFrame) -> pd.DataFrame:
    tokenizer = get_tokenizer()
    max_tokens = int(settings.model.max_tokens)
    rows: list[dict] = []

    for _, row in ingested.iterrows():
        sentences = _sentence_split(row[Article.full_text])
        span_idx = 0
        for sentence in sentences:
            for piece in _split_on_clauses(sentence, tokenizer, max_tokens):
                n_tok = _n_tokens(tokenizer, piece)
                assert n_tok <= max_tokens, f"span exceeds {max_tokens} tokens: {n_tok}"
                rows.append(
                    {
                        Article.id: row[Article.id],
                        Article.ticker: row[Article.ticker],
                        Span.id: f"{row[Article.id]}:{row[Article.ticker]}:{span_idx}",
                        Span.text: piece,
                        Span.position_index: span_idx,
                        Span.n_tokens: n_tok,
                        Span.relevant: _is_relevant(
                            piece, row[Article.ticker], row.get("name", "")
                        ),
                    }
                )
                span_idx += 1

    out = pd.DataFrame(rows)
    log.info(
        "segmented into %d spans (%d relevant)",
        len(out),
        int(out[Span.relevant].sum()) if len(out) else 0,
    )
    return out
