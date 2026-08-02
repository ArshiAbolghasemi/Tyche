# Tyche

Tyche is a two-stage research pipeline that turns raw financial news and market
data into backtested portfolio allocations.

1. **News-sentiment extraction** — an agentic pipeline (LangGraph + Azure OpenAI)
   ingests raw news, summarizes each article, scores financial sentiment, and
   neutralizes systematic bias to produce a clean per-(article, ticker) sentiment
   contract.
2. **Portfolio construction** — a multimodal deep-learning model fuses daily
   OHLCV and news-sentiment features into a predicted
   uncertainty-aware return distribution (mean plus aleatoric and epistemic covariance)
   per rebalance date. Gaussian and Student-$t$ models train solely with their
   distributional negative log-likelihood; MC dropout estimates model uncertainty at
   inference. Those forecasts
   drive six allocation strategies (EW, BL, Bayesian BL, MVO, RP, HRP), which are
   backtested with transaction costs and slippage across holding periods and cost
   scenarios.

## Setup

1. Install [`uv`](https://docs.astral.sh/uv/) and Python 3.10.
2. Install dependencies: `uv sync` (add a hardware extra if training the model,
   e.g. `uv sync --extra cpu`, `--extra mps`, or a `cu11x`/`cu12x`/`cu13x` variant).
3. Copy `.env.example` to `.env` and fill in the required values (Azure OpenAI
   key for the news scorer; everything else has a working default).
4. Pull tracked data and benchmark artifacts with DVC: `uv run dvc pull`.

Full walkthrough: [`docs/setup.md`](docs/setup.md).

## Documentation

| Section | Description |
| --- | --- |
| [Setup](docs/setup.md) | Environment, dependencies, `.env`, DVC-tracked data |
| [News Sentiment Pipeline](docs/news-pipeline.md) | The agent DAG that turns raw news into a sentiment contract |
| [Data & Features](docs/data-features.md) | Universe, calendar, feature branches, windowing/splits |
| [Predictive Model](docs/model.md) | The multimodal return-distribution network, training, and inference |
| [Portfolio Management](docs/portfolio-management.md) | Allocation strategies, transaction costs, and the backtest engine |
| [Configuration Reference](docs/configuration.md) | Every config surface, grouped by domain |

## License

MIT — see [`LICENSE`](LICENSE).
