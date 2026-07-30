# Testing

The repository doesn't yet ship an automated `pytest` suite (`pytest` is
present as a dev dependency for when one is added). Correctness is instead
verified through static checks, the pipelines' built-in audit modes, and
bounded smoke runs of each pipeline. This page describes the procedure for
each.

## Static checks

```bash
uv run ruff check .
```

Run this before any change is considered done.

## News pipeline audits

The news pipeline has a dedicated auditor agent with four modes; three are
exposed as CLI subcommands and are the primary way to validate a change to the
ingestion/summarization/scoring/neutralization stages without a full run.

```bash
# Audit A — label-order + sanity-sentence check (also runs automatically at the
# start of every `tyche run`; halts the pipeline on failure)
uv run tyche audit-a

# Audit B — (re)build the entity-prior artifact consumed by the neutralizer;
# offline/quarterly, run over a representative slice of the source data
uv run tyche audit-b --input data/zanista/news.parquet --limit 5000

# Audit C — causality check: verifies that appending future rows never changes
# past scores
uv run tyche audit-c --input data/zanista/news.parquet --limit 200
```

## News pipeline smoke run

Run the full agent graph end to end on a small, bounded slice of the source
file to confirm the ingest → summarize → score → neutralize → audit chain is
wired correctly:

```bash
uv run tyche run --input data/zanista/news.parquet --limit 50 --output /tmp/tyche_smoke.parquet
```

A bounded `--limit` run skips the LangGraph DAG and pushes the slice through
the same agents directly (see [News Sentiment Pipeline](news-pipeline.md)), so
it exercises the same code paths as a production run at a fraction of the
cost.

## Portfolio pipeline smoke run

Train/predict/backtest for a single holding period, which is the fastest way
to confirm the data-assembly, model, and allocation stages all work together:

```bash
uv run python -m tyche.portfolio.run --holding 5
```

For a broader check across the full holding-period grid (and, optionally,
multiple transaction-cost scenarios), use the wrapper script:

```bash
./scripts/run_portfolio.sh 5 20        # only these holding periods
COST_BPS="0 5 10" ./scripts/run_portfolio.sh
```

Each run logs the model-quality table (MAE/RMSE/IC/NLL/coverage) and the
per-strategy portfolio metrics table, and writes artifacts under
`benchmark/<target_distribution>/` for later inspection or report generation.

## Report generation as a validation step

Once benchmark artifacts exist for a target distribution, regenerating the
LaTeX tables is a cheap way to confirm the artifact schema hasn't drifted:

```bash
uv run python scripts/generate_portfolio_report.py
```

This only reads existing CSVs under `benchmark/`; it never re-runs training or
backtests. See [Evaluation & Reporting](evaluation-reporting.md).
