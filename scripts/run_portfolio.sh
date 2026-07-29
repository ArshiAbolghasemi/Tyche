#!/usr/bin/env bash
#
# Run the portfolio pipeline across every holding period and collate the results.
#
# The sweep runs in a single process (tyche.portfolio.sweep) rather than one process
# per H, so the data is assembled and aligned once instead of eight times — building
# the intraday tensor dominates startup. Per-H artifacts and the combined tables are
# written to tyche/portfolio/artifacts.
#
#   ./scripts/run_portfolio.sh                 # default sweep: 1 2 3 5 10 20 40 60
#   ./scripts/run_portfolio.sh 5 20 60         # only these holding periods
#   EPOCHS=5 ./scripts/run_portfolio.sh        # short run, for smoke-testing changes

set -euo pipefail

holdings=("$@")
if [ ${#holdings[@]} -eq 0 ]; then
    holdings=(1 2 3 5 10 20 40 60)
fi

args=(--holdings "${holdings[@]}")
[ -n "${EPOCHS:-}" ] && args+=(--epochs "$EPOCHS")
[ -n "${LOOKBACK:-}" ] && args+=(--lookback "$LOOKBACK")

echo "Running portfolio sweep for holding periods: ${holdings[*]}"
uv run python -m tyche.portfolio.sweep "${args[@]}"
