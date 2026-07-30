# Configuration Reference

Configuration is split by domain, each owning its own settings. Both are
exposed together through `tyche.common.config.config` (`config.news`,
`config.portfolio`).

## News pipeline (`tyche/news/config.py`)

Env-var-backed, frozen dataclasses read at access time (via
`tyche.common.env`), so `settings.model.name`,
`settings.neutralizer.rolling_window_days`, etc. always reflect the live
environment. `.env` is loaded once on import (`tyche/common/env.py`).

Sections and their env-var prefixes (see `.env.example` for the full list
with defaults):

| Section | Prefix | Covers |
| --- | --- | --- |
| Paths | `TYCHE_PATHS_*` | Input news file, output sentiment parquet |
| Ingest | `TYCHE_INGEST_*` | Grouping columns, masked-entity placeholder |
| Summarizer | `TYCHE_SUMMARIZER_*` | Device, batch size, summary length bounds |
| Embedder | `TYCHE_EMBEDDING_*` | Model name/revision, device, batch size, max tokens |
| Deduplicator | `TYCHE_DEDUP_*` | Distance threshold, clustering window |
| Sentiment scorer | `TYCHE_SENTIMENT_*` | Azure OpenAI key/endpoint/deployment/API version, retries, timeout, concurrency, optional system-prompt override |
| Neutralizer | `TYCHE_NEUTRALIZER_*` | Entity-prior path, rolling window, shrinkage, winsorization, group-size floor |
| Auditor | `TYCHE_AUDITOR_*` | Baseline path, PSI threshold, same-sign alert threshold, sanity sentences |
| Dask | `TYCHE_DASK_*` | Block size, partition count |

`TYCHE_ENV` selects the deployment profile (`development` / `staging` /
`production`); `HF_TOKEN` is optional and only needed for gated/private
HuggingFace models.

## Portfolio pipeline (`tyche/portfolio/config.py`)

Frozen dataclasses composed into one `Config`, constructed via
`default_config()` — not environment-variable-backed; overrides are applied
programmatically (e.g. the portfolio CLI's `--holding`/`--lookback`/
`--transaction-cost-bps` flags call `dataclasses.replace` on the relevant
section).

| Section | Dataclass | Covers |
| --- | --- | --- |
| Paths | `Paths` | Source parquet locations, artifacts root (`benchmark/`, split per target distribution), news-embedding cache |
| Split | `SplitConfig` | Chronological in-sample/test date ranges, train fraction |
| Window | `WindowConfig` | Lookback length `T`, holding/horizon `H`, embargo |
| Intraday | `IntradayConfig` | Resample frequency, session bounds, max bars/day |
| Daily features | `DailyFeatureConfig` | Rolling windows for volatility/momentum/RSI/ATR/z-score |
| News features | `NewsFeatureConfig` | Story-clustering lookback, similarity threshold, device, batch size |
| Model | `ModelConfig` | Conv channels/kernel, dropout, hidden dim, covariance rank/eps, sequence encoder choice |
| Train | `TrainConfig` | Epochs, batch size, LR, target distribution (`gaussian`/`student_t`), MC-dropout samples, loss weights, early-stopping patience, seed, device |
| Portfolio | `PortfolioConfig` | BL tau/risk-aversion, covariance shrinkage, max weight, turnover penalty, transaction cost/slippage bps, cost model, risk-parity solver tolerance, HRP linkage |

`Config.artifacts_dir` derives the per-run output directory from
`paths.artifacts / train.target_distribution` (e.g. `benchmark/student_t`).

To run with a modified config from Python:

```python
from dataclasses import replace
from tyche.portfolio.config import default_config

cfg = default_config()
cfg = replace(cfg, train=replace(cfg.train, target_distribution="gaussian"))
```

The CLI (`tyche.portfolio.run`) only exposes holding period(s), lookback, and
transaction-cost scenario(s) as flags; anything else requires either editing
`tyche/portfolio/config.py`'s defaults or calling the runner functions
directly with a modified `Config`.
