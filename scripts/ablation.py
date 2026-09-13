"""Run fixed-2025 ablations and write compact PDF comparisons."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tyche.ablation.experiment import (
    EXPERIMENTS,
    REGIMES,
    artifact_config,
    run_one,
    self_check,
)
from tyche.portfolio.config import default_config
from tyche.portfolio.run import config_for_holding

PORTFOLIO_METHODS = ("EW", "BL", "Bayesian_BL", "MVO", "RP", "HRP")
DISTRIBUTIONS = ("student_t", "gaussian")


def plot_metrics(metrics: pd.DataFrame, path: Path) -> None:
    portfolio = metrics[metrics["strategy"] != "predictive_model"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("sharpe", "cum_return_net", "max_drawdown"),
        ("Sharpe", "Net return", "Maximum drawdown"),
        strict=True,
    ):
        table = portfolio.pivot_table(
            index=["regime", "experiment"], columns="strategy", values=metric
        )
        table.plot.bar(ax=ax, width=0.85, legend=metric == "sharpe")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=70, labelsize=7)
        if metric == "sharpe":
            ax.legend(loc="upper left", fontsize=7)
        if metric != "sharpe" and ax.get_legend():
            ax.get_legend().remove()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_curves(curves: dict[tuple[str, str], pd.DataFrame], output: Path) -> None:
    regimes = list(dict.fromkeys(regime for regime, _ in curves))
    for strategy in PORTFOLIO_METHODS:
        for regime in regimes:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for (candidate, experiment), frame in curves.items():
                if candidate == regime and strategy in frame:
                    ax.plot(
                        frame.index,
                        frame[strategy],
                        label=experiment,
                        linewidth=1.2,
                    )
            if not ax.lines:
                plt.close(fig)
                continue
            ax.set_title(f"{strategy} · {regime}")
            ax.set_ylabel("Portfolio value")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left", ncol=2, fontsize=7)
            fig.tight_layout()
            fig.savefig(
                output / f"equity_curves_{strategy}_{regime}.pdf",
                format="pdf",
                bbox_inches="tight",
            )
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS)
    )
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=["unfiltered"])
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=DISTRIBUTIONS,
        default=list(DISTRIBUTIONS),
    )
    parser.add_argument("--holding", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmark/ablation"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        self_check()
        print("ablation self-check passed")
        return

    base = config_for_holding(default_config(), args.holding)
    for distribution in args.distributions:
        distribution_output = args.output / distribution
        distribution_output.mkdir(parents=True, exist_ok=True)
        distribution_cfg = replace(
            base, train=replace(base.train, target_distribution=distribution)
        )
        frames, curves = [], {}
        for regime in args.regimes:
            for experiment in args.experiments:
                # The macro-only control is the pure-beta rule with equal weights.
                # Run it once; every requested regime would otherwise be identical.
                if experiment == "macro_only" and regime != args.regimes[0]:
                    continue
                effective_regime = "pure_beta" if experiment == "macro_only" else regime
                cfg = artifact_config(
                    distribution_cfg, args.output, effective_regime, experiment
                )
                metric, curve = run_one(cfg, experiment)
                metric.insert(0, "regime", effective_regime)
                metric.insert(0, "distribution", distribution)
                frames.append(metric)
                curves[(effective_regime, experiment)] = curve
                run_dir = cfg.artifacts_dir
                run_dir.mkdir(parents=True, exist_ok=True)
                metric.to_csv(run_dir / "metrics.csv", index=False)
                curve.to_csv(run_dir / "equity_curves.csv")

        metrics = pd.concat(frames, ignore_index=True)
        metrics.to_csv(distribution_output / "ablation_metrics.csv", index=False)
        plot_metrics(metrics, distribution_output / "ablation_metrics.pdf")
        plot_curves(curves, distribution_output)
    print(f"wrote results and PDFs to {args.output}")


if __name__ == "__main__":
    main()
