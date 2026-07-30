# Setup

## Prerequisites

- Python 3.10 (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for dependency management and running commands
- Access to the project's DVC remote (`s3://zanista`, Cloudflare R2-backed) for
  tracked data and benchmark artifacts

## 1. Install dependencies

```bash
uv sync
```

The model-training path needs `torch`/`torchvision`, which are gated behind
hardware-specific optional extras (see `[project.optional-dependencies]` in
`pyproject.toml`):

```bash
uv sync --extra cpu     # CPU-only, any platform
uv sync --extra mps     # Apple Silicon
uv sync --extra cu118   # CUDA 11.8 (Pascal/Volta/Turing)
uv sync --extra cu126   # CUDA 12.6
uv sync --extra cu128   # CUDA 12.8
uv sync --extra cu130   # CUDA 13.0 (Blackwell)
```

Only one hardware extra may be active at a time (they're declared as mutually
exclusive in `[tool.uv.conflicts]`).

## 2. Configure environment variables

```bash
cp .env.example .env
```

Every setting has a working default except the Azure OpenAI credential used by
the news-sentiment scorer:

- `TYCHE_SENTIMENT_API_KEY` — required to run the news pipeline's scoring stage
- `HF_TOKEN` — optional; only needed for gated/private HuggingFace models or a
  higher rate limit on model revision lookups (the summarizer and embedder load
  public weights locally, no token needed by default)

See [Configuration Reference](configuration.md) for the full variable list and
how each domain reads it.

## 3. Pull tracked data and artifacts

Data (`data/`) and benchmark results (`benchmark/`) are DVC-tracked, not
committed to git.

```bash
uv run dvc pull
```

This retrieves the EODHD daily/intraday OHLCV parquet files, the news parquet
sources, and (optionally) the tracked benchmark CSV/`.npz` outputs used for
report generation.

## 4. Verify the install

```bash
uv run ruff check .
uv --version
uv run tyche --help
uv run python -m tyche.portfolio.run --help
```

Continue to [Testing](testing.md) to run the pipelines end to end on bounded
inputs.
