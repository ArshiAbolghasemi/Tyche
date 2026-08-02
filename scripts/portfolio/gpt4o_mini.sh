#!/usr/bin/env bash
#
# Portfolio run on gpt-4o-mini sentiment -> benchmark/gpt4o_mini/<distribution>/
# Needs ./scripts/news_gpt4o_mini.sh to have run first.
# Extra args override the defaults, e.g. `./scripts/portfolio_gpt4o_mini.sh --holdings 5 20`

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TYCHE_SENTIMENT_BACKENDS=gpt4o_mini \
TYCHE_PORTFOLIO_PATHS_NEWS_SENTIMENT=data/output/news_sentiment_gpt4o_mini.parquet \
  uv run python -m tyche.portfolio.run \
    --holdings 1 2 3 5 10 20 40 60 \
    --transaction-cost-bps 0 1 2 5 10 "$@"
