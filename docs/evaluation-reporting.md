# Evaluation & Reporting

## Predictive-model metrics (`tyche/portfolio/evaluation/model_metrics.py`)

Computed from saved predictions vs. realized forward returns:

- **Point-forecast quality** — MAE, RMSE, directional accuracy
- **Cross-sectional ranking skill** — Pearson IC, Spearman rank IC (the
  allocator only ever consumes relative views across assets, so ranking skill
  matters more than point accuracy)
- **Distributional fit** — Gaussian or Student-t NLL, matching
  `TrainConfig.target_distribution`
- **Calibration** — predictive-interval coverage

Written per holding period to
`benchmark/<target_distribution>/model_metrics_H<H>.csv`.

## Portfolio metrics (`tyche/portfolio/evaluation/portfolio_metrics.py`)

Computed from a backtest result, assuming daily periodicity (252 trading days
for annualization):

- **Return/risk** — cumulative gross & net return, annualized return &
  volatility, Sharpe, Sortino, Calmar, max drawdown
- **Trading behaviour** — average turnover, total costs, hit rate

Written per `(holding, [cost])` combination to
`benchmark/<target_distribution>/portfolio_metrics_[C<cost>_]H<H>.csv`, plus a
combined table (`portfolio_metrics.csv` or, when sweeping cost scenarios,
`cost_portfolio_metrics.csv`) and the per-strategy equity curves
(`equity_curves_[C<cost>_]H<H>.csv`).

## LaTeX report generation (`scripts/generate_portfolio_report.py`)

Presentation-only: reads the tracked benchmark CSVs and never re-runs
training or backtests. Run from the repository root once benchmark artifacts
exist for the distributions/holdings/costs it expects:

```bash
uv run python scripts/generate_portfolio_report.py
```

It produces three LaTeX table files under `report/generated/`:

- `model_metrics.tex` — predictive-model metrics by holding period and target
  distribution, bolding the better distribution per metric within each
  holding period
- `portfolio_results.tex` — full portfolio-metrics comparison across
  distributions, models, and holding periods, one table per transaction-cost
  scenario, bolding the best distribution/model pair per holding period and
  metric
- `best_models.tex` — the Sharpe-maximizing distribution/model pair per
  `(cost, holding)` combination, with its accompanying performance and
  turnover

These feed directly into `report/technical_report.tex`, which also embeds the
figures under `report/figures/`.
