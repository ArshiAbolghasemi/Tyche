"""End-to-end orchestrator: data -> model -> predictions -> portfolio -> comparison.

Runs the full experiment for one holding period H: prepares the aligned/standardized
data and chronological split, trains the proposed model plus the two ablations
(no-news, no-intraday), saves out-of-sample predictions, evaluates the predictive
model, then backtests the proposed strategy against equal-weight, historical MVO,
historical BL, and the no-BL / ablation variants under one identical universe, cost,
and rebalance schedule. Prints two comparison tables and saves artifacts.

Since priced data ends 2024-12-31, the validation window doubles as the out-of-sample
test; point ``SplitConfig.test_*`` at a real 2025 range once those prices exist.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from tyche.portfolio.data.assemble import assemble
from tyche.portfolio.allocation.backtest import run_backtest
from tyche.portfolio.config import Config, default_config
from tyche.portfolio.evaluation.model_metrics import evaluate_model
from tyche.portfolio.evaluation.portfolio_metrics import evaluate_portfolio
from tyche.portfolio.model.predict import Predictions, predict
from tyche.portfolio.data.preprocessing import apply_standardizer, fit_standardizer
from tyche.portfolio.allocation.strategies import (
    equal_weight,
    historical_bl,
    historical_mvo,
    model_bl,
    model_no_bl,
)
from tyche.portfolio.model.train import train_model
from tyche.portfolio.data.windows import SplitIndex, build_splits, train_day_indices


def prepare(cfg: Config):
    data = assemble(cfg)
    splits = build_splits(data, cfg)
    std = apply_standardizer(
        data, fit_standardizer(data, train_day_indices(splits, cfg))
    )
    oos = splits.test if splits.test else splits.val
    return data, std, splits, oos


def _train_and_predict(std, splits: SplitIndex, oos, cfg, **flags) -> Predictions:
    result = train_model(std, splits.train, splits.val, cfg, **flags)
    return predict(result.model, std, oos)


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
    data, std, splits, oos = prepare(cfg)
    print(
        f"universe={data.assets} | train={len(splits.train)} val={len(splits.val)} "
        f"oos={len(oos)} | H={cfg.window.holding} T={cfg.window.lookback}"
    )

    preds = {
        "proposed": _train_and_predict(std, splits, oos, cfg),
        "no_news": _train_and_predict(std, splits, oos, cfg, use_news=False),
        "no_intraday": _train_and_predict(std, splits, oos, cfg, use_intraday=False),
    }
    model_metrics = {k: evaluate_model(v) for k, v in preds.items()}

    cfg.paths.artifacts.mkdir(parents=True, exist_ok=True)
    for name, p in preds.items():
        p.save(cfg.paths.artifacts / f"pred_{name}_H{cfg.window.holding}.npz")

    rebal_t = _rebalance_days(oos, cfg)
    strategies = {
        "proposed": model_bl(preds["proposed"], data.adj_close, cfg),
        "equal_weight": equal_weight(data.n_assets),
        "historical_mvo": historical_mvo(data.adj_close, cfg),
        "historical_bl": historical_bl(data.adj_close, cfg),
        "no_bl": model_no_bl(preds["proposed"], data.adj_close, cfg),
        "no_news": model_bl(preds["no_news"], data.adj_close, cfg),
        "no_intraday": model_bl(preds["no_intraday"], data.adj_close, cfg),
    }
    port_metrics, curves = {}, {}
    for name, strat in strategies.items():
        net, metrics = _backtest(data, data.days, rebal_t, strat, cfg)
        port_metrics[name] = metrics
        curves[name] = pd.Series(net.value, index=net.dates)

    return {
        "model_metrics": model_metrics,
        "portfolio_metrics": port_metrics,
        "curves": curves,
    }


def _print_table(title: str, metrics: dict[str, dict[str, float]]) -> None:
    df = pd.DataFrame(metrics).T
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Multimodal portfolio pipeline")
    ap.add_argument("--holding", type=int, default=None, help="H (holding period)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lookback", type=int, default=None)
    args = ap.parse_args()

    cfg = default_config()
    if args.holding is not None:
        cfg = replace(
            cfg,
            window=replace(
                cfg.window,
                holding=args.holding,
                embargo=max(cfg.window.embargo, args.holding),
            ),
        )
    if args.lookback is not None:
        cfg = replace(cfg, window=replace(cfg.window, lookback=args.lookback))
    if args.epochs is not None:
        cfg = replace(cfg, train=replace(cfg.train, epochs=args.epochs))

    results = run_experiment(cfg)
    _print_table("Predictive model", results["model_metrics"])
    _print_table("Portfolio backtest", results["portfolio_metrics"])


if __name__ == "__main__":
    main()
