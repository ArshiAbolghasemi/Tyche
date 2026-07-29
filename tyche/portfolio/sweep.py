"""Run the full experiment across a range of holding periods and collate the results.

``run.py`` answers "how did this configuration do?"; this answers "how does the answer
move with H?" — which is the question the transaction-cost model makes interesting.
Shorter holding periods rebalance more often and pay the round-trip charge more often,
so the H sweep is where the cost model earns its keep: the gross ranking of strategies
and the net ranking need not agree, and where they diverge is the useful finding.

Each H is a fully independent experiment — its own embargo, its own trained models, its
own predictions — so results are written per H as well as combined, and a failure in
one H does not lose the others.

Outputs, all under ``Paths.artifacts``:

* ``portfolio_metrics_H{h}.csv`` / ``model_metrics_H{h}.csv`` — one table per H
* ``sweep_portfolio_metrics.csv`` / ``sweep_model_metrics.csv`` — every H stacked,
  indexed by (holding, strategy)
* ``equity_curves_H{h}.csv`` — daily value series per strategy
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from tyche.common.logging import get_logger
from tyche.portfolio.config import Config, default_config
from tyche.portfolio.run import run_experiment

log = get_logger(__name__)

DEFAULT_HOLDINGS: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 60)


def config_for_holding(cfg: Config, holding: int) -> Config:
    """Config for one holding period, with the embargo widened to match.

    The embargo has to be at least H or a training sample's forward window can overlap
    the validation split, so it is derived rather than left to the caller."""
    return replace(
        cfg,
        window=replace(
            cfg.window, holding=holding, embargo=max(cfg.window.embargo, holding)
        ),
    )


def _tidy(metrics: dict[str, dict[str, float]], holding: int) -> pd.DataFrame:
    df = pd.DataFrame(metrics).T
    df.index.name = "strategy"
    return df.reset_index().assign(holding=holding).set_index(["holding", "strategy"])


def run_sweep(
    cfg: Config, holdings: tuple[int, ...] = DEFAULT_HOLDINGS
) -> dict[str, pd.DataFrame]:
    """Run every holding period, writing per-H artifacts and returning the collation."""
    cfg.paths.artifacts.mkdir(parents=True, exist_ok=True)
    portfolio_frames: list[pd.DataFrame] = []
    model_frames: list[pd.DataFrame] = []
    failed: dict[int, str] = {}

    for holding in holdings:
        log.info("=== holding period H=%d ===", holding)
        try:
            results = run_experiment(config_for_holding(cfg, holding))
        except Exception as exc:  # one H failing must not lose the rest of the sweep
            log.exception("H=%d failed: %s", holding, exc)
            failed[holding] = str(exc)
            continue

        portfolio = _tidy(results["portfolio_metrics"], holding)
        model = _tidy(results["model_metrics"], holding)
        portfolio_frames.append(portfolio)
        model_frames.append(model)

        portfolio.to_csv(cfg.paths.artifacts / f"portfolio_metrics_H{holding}.csv")
        model.to_csv(cfg.paths.artifacts / f"model_metrics_H{holding}.csv")
        pd.DataFrame(results["curves"]).to_csv(
            cfg.paths.artifacts / f"equity_curves_H{holding}.csv"
        )

    if not portfolio_frames:
        raise RuntimeError(f"every holding period failed: {failed}")

    combined = {
        "portfolio": pd.concat(portfolio_frames).sort_index(),
        "model": pd.concat(model_frames).sort_index(),
    }
    combined["portfolio"].to_csv(cfg.paths.artifacts / "sweep_portfolio_metrics.csv")
    combined["model"].to_csv(cfg.paths.artifacts / "sweep_model_metrics.csv")

    if failed:
        log.warning("holding periods that failed: %s", sorted(failed))
    return combined


def _print_pivot(df: pd.DataFrame, metric: str) -> None:
    if metric not in df.columns:
        return
    table = df[metric].unstack("strategy")
    print(f"\n=== {metric} by holding period ===")
    with pd.option_context(
        "display.width",
        200,
        "display.max_columns",
        None,
        "display.float_format",
        lambda x: f"{x:.4f}",
    ):
        print(table)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep the pipeline over holding periods")
    ap.add_argument(
        "--holdings",
        type=int,
        nargs="+",
        default=list(DEFAULT_HOLDINGS),
        help="holding periods to run (default: 1 2 3 5 10 20 40 60)",
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lookback", type=int, default=None)
    args = ap.parse_args()

    cfg = default_config()
    if args.lookback is not None:
        cfg = replace(cfg, window=replace(cfg.window, lookback=args.lookback))
    if args.epochs is not None:
        cfg = replace(cfg, train=replace(cfg.train, epochs=args.epochs))

    combined = run_sweep(cfg, tuple(args.holdings))
    for metric in ("cum_return_net", "sharpe", "max_drawdown", "avg_turnover"):
        _print_pivot(combined["portfolio"], metric)
    print(f"\nartifacts written to {cfg.paths.artifacts}")


if __name__ == "__main__":
    main()
