#!/usr/bin/env bash
#
# Alpha-beta arm: portfolio run on finbert sentiment, gaussian target
# distribution, with macro-beta stock filtering.
#   -> benchmark_macro_beta/finbert/gaussian/
#
# Pairs with the baseline in scripts/portfolio/gaussian/finbert.sh and the I-MACD
# arm in scripts/portfolio/imacd/gaussian/finbert.sh. All three train the same
# model on the same features; they differ only in which names the allocator may
# hold. Compare portfolio metrics across the three to judge the filters.
#
# Needs ./scripts/news/finbert.sh and
# ./scripts/fetch_macro_indicators.py to have run first.
#
# Extra args override the defaults, e.g.
#   ./scripts/portfolio/macro_beta/gaussian/finbert.sh --holdings 40 60
# Env knobs: STRATEGY, Z_THRESHOLD, MASK_MODE, HOLD_DAYS, USE_BETA_SIGN

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

TYCHE_SENTIMENT_BACKENDS=finbert \
TYCHE_PORTFOLIO_PATHS_NEWS_SENTIMENT=data/output/news_sentiment_finbert.parquet \
TYCHE_PORTFOLIO_TRAIN_TARGET_DISTRIBUTION=gaussian \
TYCHE_PORTFOLIO_ALPHA_FILTER_ENABLED=true \
TYCHE_PORTFOLIO_ALPHA_FILTER_INDICATOR=macro_beta \
TYCHE_PORTFOLIO_ALPHA_FILTER_MASK_MODE=${MASK_MODE:-buy_only} \
TYCHE_PORTFOLIO_MACRO_BETA_STRATEGY=${STRATEGY:-own_minus_ind} \
TYCHE_PORTFOLIO_MACRO_BETA_Z_THRESHOLD=${Z_THRESHOLD:-2.0} \
TYCHE_PORTFOLIO_MACRO_BETA_HOLD_DAYS=${HOLD_DAYS:-0} \
TYCHE_PORTFOLIO_MACRO_BETA_USE_BETA_SIGN=${USE_BETA_SIGN:-false} \
TYCHE_PORTFOLIO_PATHS_ARTIFACTS=benchmark_macro_beta \
  uv run python -m tyche.portfolio.run \
    --holdings 1 2 3 5 10 20 40 60 \
    --transaction-cost-bps 0 1 2 5 10 20 50 100 "$@"
