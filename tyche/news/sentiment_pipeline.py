"""Tyche CLI + pipeline entrypoint — agentic FinBERT news-sentiment extraction.

    uv run python -m tyche.news.sentiment_pipeline run  [--input PATH] [--output PATH] [--limit N]
    uv run python -m tyche.news.sentiment_pipeline audit-a
    uv run python -m tyche.news.sentiment_pipeline audit-b [--input PATH] [--limit N]
    uv run python -m tyche.news.sentiment_pipeline audit-c [--input PATH] [--limit N]

Also runnable as the console script installed by ``[project.scripts]``: ``uv run tyche ...``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from tyche.common.config import settings
from tyche.common.logging import configure_logging, get_logger
from tyche.news.agents import (
    aggregator,
    auditor,
    ingest,
    neutralizer,
    scorer,
    segmentation,
)
from tyche.news.graph import build_graph
from tyche.news.records import OUTPUT_COLUMNS

log = get_logger("tyche.main")


def _ingest_limited(input_path: Optional[str], limit: Optional[int]) -> pd.DataFrame:
    # Cap the source read so the multi-GB feed is never fully loaded; limit is the
    # number of source articles (a row explodes into one row per ticker).
    return ingest.ingest(input_path, nrows=limit)


def _write_contract(
    neutralized: pd.DataFrame, output_path: Optional[str]
) -> pd.DataFrame:
    contract = neutralized[
        [c for c in OUTPUT_COLUMNS if c in neutralized.columns]
    ].copy()
    out_path = Path(str(output_path or settings.paths.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    contract.to_parquet(out_path, index=False)
    log.info("wrote %d rows to %s", len(contract), out_path)
    return contract


def run(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Score a news file end-to-end. Returns the contract frame and writes parquet."""
    auditor.audit_a()  # startup guard — halts before scoring if the model is wrong

    if limit:
        # Bounded run: ingest a slice directly, then push it through the same
        # agents the graph would use (skips re-reading the full feed via state).
        ingested = _ingest_limited(input_path, limit)
        aggregated = aggregator.aggregate(scorer.score(segmentation.segment(ingested)))
        neutralized = neutralizer.neutralize(aggregated)
        auditor.audit_d(neutralized)
    else:
        graph = build_graph()
        final_state = graph.invoke({"input_path": input_path})
        neutralized = final_state["neutralized"]

    return _write_contract(neutralized, output_path)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Tyche FinBERT sentiment pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="score a news file end-to-end")
    p_run.add_argument("--input", default=None)
    p_run.add_argument("--output", default=None)
    p_run.add_argument("--limit", type=int, default=None)

    sub.add_parser("audit-a", help="verify label order + sanity sentences")

    for name, helptext in [
        ("audit-b", "build entity_prior artifact"),
        ("audit-c", "causality verification"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--input", default=None)
        p.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.command == "run":
        run(args.input, args.output, args.limit)
    elif args.command == "audit-a":
        auditor.audit_a()
        log.info("audit-a OK")
    elif args.command == "audit-b":
        auditor.audit_b(_ingest_limited(args.input, args.limit))
    elif args.command == "audit-c":
        ingested = _ingest_limited(args.input, args.limit)
        aggregated = aggregator.aggregate(scorer.score(segmentation.segment(ingested)))
        auditor.audit_c(aggregated)
        log.info("audit-c OK")


if __name__ == "__main__":
    main()
