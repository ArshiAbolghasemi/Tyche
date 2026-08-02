#!/usr/bin/env bash
#
# Portfolio run on FinBERT sentiment -> benchmark/finbert/<distribution>/
# Needs ./scripts/news_finbert.sh to have run first.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TYCHE_SENTIMENT_BACKENDS=finbert \
TYCHE_PORTFOLIO_PATHS_NEWS_SENTIMENT=data/output/news_sentiment_finbert.parquet \
  uv run python -m tyche.portfolio.run \
    --holdings 1 2 3 5 10 20 40 60 \
    --transaction-cost-bps 0 1 2 5 10 "$@"
