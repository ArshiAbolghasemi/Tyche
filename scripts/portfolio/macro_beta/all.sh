#!/usr/bin/env bash
#
# Run every alpha-beta portfolio job: all sentiment backends x both target
# distributions. Extra arguments are forwarded to each backend script.
#
# Examples:
#   ./scripts/portfolio/macro_beta/all.sh
#   ./scripts/portfolio/macro_beta/all.sh --holdings 40 60
#   BACKENDS="gpt4o_mini finbert" DISTRIBUTIONS=gaussian ./scripts/portfolio/macro_beta/all.sh
#   STRATEGY=intersection_own_ind ./scripts/portfolio/macro_beta/all.sh --holdings 40

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BACKENDS="${BACKENDS:-finbert gpt4o_mini llama2 mistral}"
DISTRIBUTIONS="${DISTRIBUTIONS:-gaussian student_t}"

failed=()
for dist in $DISTRIBUTIONS; do
  for backend in $BACKENDS; do
    script="./$dist/$backend.sh"
    if [[ ! -x "$script" ]]; then
      echo "skipping $script (not found or not executable)" >&2
      failed+=("$dist/$backend: missing")
      continue
    fi

    echo "=== $dist / $backend ==="
    if ! "$script" "$@"; then
      echo "FAILED: $dist/$backend" >&2
      failed+=("$dist/$backend")
    fi
  done
done

if (("${#failed[@]}")); then
  printf 'failed runs: %s\n' "${failed[*]}" >&2
  exit 1
fi

echo "all alpha-beta runs complete -> benchmark_macro_beta/"
