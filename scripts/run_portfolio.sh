#!/usr/bin/env bash
#
# Run the portfolio pipeline across holding periods and collate the results.
#
# Per-H artifacts and the combined tables are written to tyche/portfolio/artifacts.
#
#   ./scripts/run_portfolio.sh                 # default sweep: 1 2 3 5 10 20 40 60
#   ./scripts/run_portfolio.sh 5 20 60         # only these holding periods
#   COST_BPS="1 2 5 10" ./scripts/run_portfolio.sh

set -euo pipefail

holdings=("$@")
if [ ${#holdings[@]} -eq 0 ]; then
    holdings=(1 2 3 5 10 20 40 60)
fi

args=(--holdings "${holdings[@]}")
[ -n "${LOOKBACK:-}" ] && args+=(--lookback "$LOOKBACK")
if [ -n "${COST_BPS:-}" ]; then
    # shellcheck disable=SC2206
    costs=(${COST_BPS})
    args+=(--transaction-cost-bps "${costs[@]}")
fi

echo "Running portfolio grid for holding periods: ${holdings[*]}"
[ -n "${COST_BPS:-}" ] && echo "Transaction cost scenarios (bps): ${COST_BPS}"
uv run python -m tyche.portfolio.run "${args[@]}"
