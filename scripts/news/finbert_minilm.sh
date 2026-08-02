#!/usr/bin/env bash
#
# Score the news feed with a MiniLM-distilled FinBERT, loaded in-process.
# No default checkpoint exists on the Hub — set NAME below (or via env) to one you
# have vetted, and make sure LABELS matches its classification-head order, or the
# scores come out silently permuted.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TYCHE_SENTIMENT_BACKENDS=finbert_minilm \
TYCHE_SENTIMENT_FINBERT_MINILM_NAME="${TYCHE_SENTIMENT_FINBERT_MINILM_NAME:-}" \
TYCHE_SENTIMENT_FINBERT_MINILM_LABELS=positive,negative,neutral \
TYCHE_SENTIMENT_FINBERT_MINILM_DEVICE=auto \
TYCHE_SENTIMENT_FINBERT_MINILM_BATCH_SIZE=64 \
  uv run python -m tyche.news.sentiment_pipeline run \
    --output data/output/news_sentiment_finbert_minilm.parquet "$@"
