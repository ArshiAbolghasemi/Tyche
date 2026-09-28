"""Sentiment ablations on the fixed chronological out-of-sample split."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tyche.portfolio.config import Config
from tyche.portfolio.data.assemble import AlignedData, assemble
from tyche.portfolio.data.preprocessing import apply_standardizer, fit_standardizer
from tyche.portfolio.data.windows import build_splits, train_day_indices
from tyche.portfolio.evaluation.model_metrics import evaluate_model
from tyche.portfolio.model.predict import predict
from tyche.portfolio.model.train import train_model
from tyche.portfolio.run import (
    PreparedExperiment,
    _build_forecasts,
    _historical_cov,
    _run_portfolios,
)

EXPERIMENTS = (
    "full",
    "no_news",
    "coverage_only",
    "constant_sentiment",
    "shuffled_sentiment",
    "article_level",
    "news_only",
    "macro_only",
)
REGIMES = ("unfiltered", "pure_alpha", "pure_beta", "beta")


def config_for_regime(cfg: Config, regime: str) -> Config:
    if regime == "unfiltered":
        return replace(
            cfg,
            daily=replace(cfg.daily, imacd_enabled=False),
            alpha_filter=replace(cfg.alpha_filter, enabled=False),
        )
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; choose from {REGIMES}")
    return replace(
        cfg,
        daily=replace(cfg.daily, imacd_enabled=False),
        alpha_filter=replace(cfg.alpha_filter, enabled=True, indicator="macro_alpha"),
        macro_alpha=replace(cfg.macro_alpha, strategy=regime),
    )


def _observed(data: AlignedData) -> np.ndarray:
    return data.news[..., data.news_names.index("log_n_articles")] > 0


def transform(data: AlignedData, cfg: Config, experiment: str) -> AlignedData:
    """Apply a placebo without using returns or future sentiment in its construction."""
    daily, news = data.daily.copy(), data.news.copy()
    sent = data.news_names.index("mean_sent")
    observed = _observed(data)
    train = data.days <= pd.Timestamp(cfg.split.in_sample_end, tz="UTC")

    if experiment in {"no_news", "macro_only"}:
        news.fill(0)
    elif experiment == "news_only":
        daily.fill(0)
    elif experiment == "coverage_only":
        news[..., sent] = 0
    elif experiment == "constant_sentiment":
        source = news[..., sent][observed & train[None, :]]
        news[..., sent] = np.where(
            observed, float(source.mean()) if source.size else 0, 0
        )
    elif experiment == "shuffled_sentiment":
        rng = np.random.default_rng(cfg.train.seed)
        for period in (train, ~train):
            mask = observed & period[None, :]
            news[..., sent][mask] = rng.permutation(news[..., sent][mask])
    elif experiment not in {"full", "article_level"}:
        raise ValueError(
            f"unknown experiment {experiment!r}; choose from {EXPERIMENTS}"
        )
    return replace(data, daily=daily, news=news)


def cov_label(experiment: str, source: str) -> str:
    return experiment if source == "total" else f"{experiment}_cov_{source}"


def run_one(
    cfg: Config, experiment: str, cov_sources: tuple[str, ...] = ("total",)
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Train once, then allocate with each covariance source on the same predictions.

    Rows and curves are labelled ``cov_label(experiment, source)``."""
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"unknown experiment {experiment!r}; choose from {EXPERIMENTS}"
        )
    if experiment == "article_level":
        cfg = replace(cfg, news=replace(cfg.news, dedup_enabled=False))

    raw = transform(assemble(cfg), cfg, experiment)
    splits = build_splits(raw, cfg)
    if not splits.train or not splits.test:
        raise RuntimeError("ablation requires non-empty train and test splits")
    standardizer = fit_standardizer(raw, train_day_indices(splits, cfg))
    data = apply_standardizer(raw, standardizer)
    use_news = experiment not in {"no_news", "macro_only"}
    trained = train_model(
        data, splits.train, splits.val, cfg, use_news=use_news, tag=experiment
    )
    predictions = predict(
        trained.model,
        data,
        splits.test,
        batch_size=cfg.train.batch_size,
        mc_dropout_samples=cfg.train.mc_dropout_samples,
        device=cfg.train.device,
    )
    frames, curves = [], {}
    for source in cov_sources:
        source_cfg = replace(cfg, portfolio=replace(cfg.portfolio, cov_source=source))
        forecasts, reference_covs = _build_forecasts(
            predictions, data.adj_close, source_cfg
        )
        prepared = PreparedExperiment(
            data=data,
            splits=splits,
            oos=splits.test,
            predictions=predictions,
            forecasts=forecasts,
            reference_covs=reference_covs,
            oos_label="test",
        )
        result = _run_portfolios(prepared, source_cfg)
        metrics = (
            pd.DataFrame(result["portfolio_metrics"])
            .T.rename_axis("strategy")
            .reset_index()
        )
        if experiment == "macro_only":
            # Macro variables are a stock-selection rule in Tyche, not model inputs.
            # EW is therefore the only honest macro-only portfolio; learned
            # allocators would add an intercept-only neural forecast to the rule.
            metrics = metrics[metrics["strategy"] == "EW"]
            result["curves"] = {"EW": result["curves"]["EW"]}
        label = cov_label(experiment, source)
        metrics.insert(0, "experiment", label)
        frames.append(metrics)
        curve = pd.DataFrame(result["curves"])
        curve.index.name = "date"
        curves[label] = curve
    model = pd.DataFrame(
        [
            evaluate_model(
                predictions, cfg.train.target_distribution, cfg.train.student_t_df
            )
        ]
    )
    model.insert(0, "experiment", experiment)
    frames.append(model.assign(strategy="predictive_model"))
    return pd.concat(frames), curves


def artifact_config(cfg: Config, root: Path, regime: str, experiment: str) -> Config:
    paths = replace(cfg.paths, artifacts=root / regime / experiment)
    return replace(config_for_regime(cfg, regime), paths=paths)


def self_check() -> None:
    days = pd.date_range("2024-12-30", periods=4, tz="UTC")
    data = AlignedData(
        assets=["A"],
        days=days,
        daily=np.ones((1, 4, 1)),
        news=np.array([[[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [0.0, 0.0]]]),
        adj_close=np.ones((1, 4)),
        daily_names=["x"],
        news_names=["mean_sent", "log_n_articles"],
    )
    cfg = Config()
    coverage = transform(data, cfg, "coverage_only")
    assert not coverage.news[..., 0].any()
    assert np.array_equal(coverage.news[..., 1], data.news[..., 1])
    constant = transform(data, cfg, "constant_sentiment")
    assert constant.news[0, 0, 0] == constant.news[0, 1, 0]
    assert transform(data, cfg, "news_only").daily.sum() == 0

    # Historical covariance at t must ignore prices after t.
    prices = np.exp(np.random.default_rng(0).normal(0, 0.01, (3, 40)).cumsum(axis=1))
    shocked = prices.copy()
    shocked[:, 31:] *= 5
    assert np.allclose(
        _historical_cov(prices, 30, cfg), _historical_cov(shocked, 30, cfg)
    )
