"""Portfolio backtest runner for BL, MVO, RP, and HRP.

The runner prepares the aligned data and chronological split, then backtests only the
four requested allocation models under a shared universe, cost model, and rebalance
schedule. It can run one holding period or a grid of holding periods and transaction
costs, writing per-run artifacts plus a combined portfolio-metrics table.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from tyche.portfolio.data.assemble import assemble
from tyche.portfolio.allocation.backtest import run_backtest
from tyche.portfolio.config import Config, default_config
from tyche.portfolio.evaluation.portfolio_metrics import evaluate_portfolio
from tyche.portfolio.allocation.strategies import (
    bl,
    mvo,
    hrp,
    rp,
)
from tyche.portfolio.data.windows import build_splits

DEFAULT_HOLDINGS: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 60)


def prepare(cfg: Config):
    data = assemble(cfg)
    splits = build_splits(data, cfg)
    oos = splits.test if splits.test else splits.val
    return data, splits, oos


def _rebalance_days(oos, cfg: Config) -> list[int]:
    ts = sorted(s.t for s in oos)
    return ts[:: cfg.window.holding]


def _backtest(data, days, rebal_t, strategy, cfg: Config):
    net = run_backtest(data.adj_close, days, rebal_t, strategy, cfg)
    zero_cost = replace(
        cfg,
        portfolio=replace(cfg.portfolio, transaction_cost_bps=0.0, slippage_bps=0.0),
    )
    gross = run_backtest(data.adj_close, days, rebal_t, strategy, zero_cost)
    return net, evaluate_portfolio(net, gross.value)


def run_experiment(cfg: Config) -> dict:
    data, splits, oos = prepare(cfg)
    oos_label = "test" if splits.test else "val (no test samples)"
    print(
        f"universe={data.assets} | train={len(splits.train)} val={len(splits.val)} "
        f"oos={len(oos)} [{oos_label}] | H={cfg.window.holding} "
        f"T={cfg.window.lookback} | cost_model={cfg.portfolio.cost_model}"
    )

    cfg.paths.artifacts.mkdir(parents=True, exist_ok=True)

    rebal_t = _rebalance_days(oos, cfg)
    strategies = {
        "BL": bl(data.adj_close, cfg),
        "MVO": mvo(data.adj_close, cfg),
        "RP": rp(data.adj_close, cfg),
        "HRP": hrp(data.adj_close, cfg),
    }
    port_metrics, curves = {}, {}
    for name, strat in strategies.items():
        net, metrics = _backtest(data, data.days, rebal_t, strat, cfg)
        port_metrics[name] = metrics
        curves[name] = pd.Series(net.value, index=net.dates)

    return {
        "portfolio_metrics": port_metrics,
        "curves": curves,
    }


def config_for_holding(cfg: Config, holding: int) -> Config:
    """Return a config for one holding period with a matching split embargo."""
    return replace(
        cfg,
        window=replace(
            cfg.window, holding=holding, embargo=max(cfg.window.embargo, holding)
        ),
    )


def config_for_transaction_cost(cfg: Config, cost_bps: float) -> Config:
    """Return a config with only the explicit transaction-cost component changed."""
    return replace(
        cfg,
        portfolio=replace(cfg.portfolio, transaction_cost_bps=float(cost_bps)),
    )


def _cost_label(cost_bps: float) -> str:
    return f"{cost_bps:g}".replace(".", "p")


def _tidy(
    metrics: dict[str, dict[str, float]], holding: int, cost_bps: float | None = None
) -> pd.DataFrame:
    df = pd.DataFrame(metrics).T
    df.index.name = "model"
    df = df.reset_index().assign(holding=holding)
    index_cols = ["holding", "model"]
    if cost_bps is not None:
        df = df.assign(transaction_cost_bps=float(cost_bps))
        index_cols = ["transaction_cost_bps", *index_cols]
    return df.set_index(index_cols)


def run_grid(
    cfg: Config,
    holdings: tuple[int, ...] = DEFAULT_HOLDINGS,
    transaction_cost_bps: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Run the requested holding/cost grid and write portfolio artifacts."""
    cfg.paths.artifacts.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    failed: dict[tuple[float, int], str] = {}
    costs = transaction_cost_bps or (cfg.portfolio.transaction_cost_bps,)
    multi_cost = len(costs) > 1

    for cost_bps in costs:
        cost_cfg = config_for_transaction_cost(cfg, cost_bps)
        for holding in holdings:
            print(f"\n=== transaction cost {cost_bps:g} bps | H={holding} ===")
            try:
                results = run_experiment(config_for_holding(cost_cfg, holding))
            except Exception as exc:
                failed[(float(cost_bps), holding)] = str(exc)
                print(f"FAILED cost={cost_bps:g} H={holding}: {exc}")
                continue

            portfolio = _tidy(
                results["portfolio_metrics"],
                holding,
                cost_bps if multi_cost else None,
            )
            frames.append(portfolio)

            suffix = (
                f"_C{_cost_label(cost_bps)}_H{holding}"
                if multi_cost
                else f"_H{holding}"
            )
            portfolio.to_csv(cfg.paths.artifacts / f"portfolio_metrics{suffix}.csv")
            pd.DataFrame(results["curves"]).to_csv(
                cfg.paths.artifacts / f"equity_curves{suffix}.csv"
            )

    if not frames:
        raise RuntimeError(f"every portfolio run failed: {failed}")

    combined = pd.concat(frames).sort_index()
    combined_name = (
        "cost_portfolio_metrics.csv" if multi_cost else "portfolio_metrics.csv"
    )
    combined.to_csv(cfg.paths.artifacts / combined_name)
    if failed:
        print(f"\nfailed runs: {sorted(failed)}")
    return combined


def _print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    with pd.option_context(
        "display.width",
        160,
        "display.max_columns",
        None,
        "display.float_format",
        lambda x: f"{x:.4f}",
    ):
        print(df)


def _print_pivot(df: pd.DataFrame, metric: str) -> None:
    if metric not in df.columns:
        return
    table = df[metric].unstack("model")
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
    ap = argparse.ArgumentParser(description="Multimodal portfolio pipeline")
    ap.add_argument("--holding", type=int, default=None, help="single H")
    ap.add_argument(
        "--holdings",
        type=int,
        nargs="+",
        default=None,
        help="holding periods to run (default: 1 2 3 5 10 20 40 60)",
    )
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument(
        "--transaction-cost-bps",
        type=float,
        nargs="+",
        default=None,
        help="transaction-cost scenarios in bps (slippage is left unchanged)",
    )
    args = ap.parse_args()

    cfg = default_config()
    if args.lookback is not None:
        cfg = replace(cfg, window=replace(cfg.window, lookback=args.lookback))
    if args.holding is not None and args.holdings is not None:
        raise SystemExit("use either --holding or --holdings, not both")

    if args.holding is not None:
        cfg = config_for_holding(cfg, args.holding)
        if args.transaction_cost_bps:
            if len(args.transaction_cost_bps) != 1:
                raise SystemExit("--holding accepts at most one transaction cost")
            cfg = config_for_transaction_cost(cfg, args.transaction_cost_bps[0])
        results = run_experiment(cfg)
        _print_table("Portfolio backtest", pd.DataFrame(results["portfolio_metrics"]).T)
        return

    holdings = tuple(args.holdings) if args.holdings else DEFAULT_HOLDINGS
    combined = run_grid(cfg, holdings, args.transaction_cost_bps)
    for metric in ("cum_return_net", "sharpe", "max_drawdown", "avg_turnover"):
        _print_pivot(combined, metric)
    print(f"\nartifacts written to {cfg.paths.artifacts}")


if __name__ == "__main__":
    main()
