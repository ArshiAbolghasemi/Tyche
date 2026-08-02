#!/usr/bin/env bash
#
# Portfolio run on Mistral-7B-Instruct sentiment
#   -> benchmark/mistral_7b_instruct/<distribution>/
# Needs ./scripts/news_mistral.sh to have run first. The vLLM container is not
# needed here (this reads the parquet), so you can stop it to free the GPU.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TYCHE_SENTIMENT_BACKENDS=mistral_7b_instruct \
TYCHE_PORTFOLIO_PATHS_NEWS_SENTIMENT=data/output/news_sentiment_mistral_7b_instruct.parquet \
  uv run python -m tyche.portfolio.run \
    --holdings 1 2 3 5 10 20 40 60 \
    --transaction-cost-bps 0 1 2 5 10 "$@"
