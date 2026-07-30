"""Portfolio-pipeline configuration — env-var-backed dataclass sections.

Mirrors the style of ``tyche.news.config``: every section is a small frozen
``@dataclass`` whose fields are read from environment variables (via
``tyche.common.env``) at *construction* time, under the ``TYCHE_PORTFOLIO_*``
prefix. ``.env`` is loaded once by ``tyche.common.env`` on import.

Unlike the news config, this module keeps ``Config`` as a plain dataclass (built
by ``default_config()``) rather than exposing a live properties-based settings
singleton: ``tyche.portfolio.run`` sweeps holding periods and transaction-cost
scenarios by calling ``dataclasses.replace()`` on a ``Config`` instance (see
``config_for_holding`` / ``config_for_transaction_cost``), which requires an
explicit dataclass instance to replace fields on.

Defaults target the data actually on disk (prices cover 2023-10-02 ->
2024-12-31, 5 assets); set a real ``test`` range via env once out-of-sample
prices exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tyche.common.env import _env, _env_list

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(key: str, default: str) -> Path:
    """Env-var path, resolved against ``REPO_ROOT`` unless already absolute."""
    return REPO_ROOT / _env(key, default)


def _env_int_tuple(key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(v) for v in _env_list(key, [str(v) for v in default]))


@dataclass(frozen=True)
class Paths:
    news_sentiment: Path = field(
        default_factory=lambda: _env_path(
            "TYCHE_PORTFOLIO_PATHS_NEWS_SENTIMENT",
            "data/output/news_sentiment.parquet",
        )
    )
    daily_ohlcv: Path = field(
        default_factory=lambda: _env_path(
            "TYCHE_PORTFOLIO_PATHS_DAILY_OHLCV", "data/eodhd/daily_ohclv.parquet"
        )
    )
    intraday_ohlcv: Path = field(
        default_factory=lambda: _env_path(
            "TYCHE_PORTFOLIO_PATHS_INTRADAY_OHLCV",
            "data/eodhd/intraday_ohclv.parquet",
        )
    )
    # Portfolio-run and model-training artifacts; ``Config.artifacts_dir`` appends the
    # target-distribution subdirectory (benchmark/student_t, benchmark/gaussian, ...).
    artifacts: Path = field(
        default_factory=lambda: _env_path("TYCHE_PORTFOLIO_PATHS_ARTIFACTS", "benchmark")
    )
    news_embedding_cache: Path = field(
        default_factory=lambda: _env_path(
            "TYCHE_PORTFOLIO_PATHS_NEWS_EMBEDDING_CACHE",
            ".cache/news_embedding_cache.npz",
        )
    )


@dataclass(frozen=True)
class SplitConfig:
    """Strictly chronological split (no shuffling). ``purge``/``embargo`` drop
    samples whose forward target window leaks across a boundary.

    The in-sample period is split ``train_fraction`` / ``1 - train_fraction`` on the
    **trading-day index** (not the calendar), so the boundary adapts to the data rather
    than being hard-coded. Everything from ``test_start`` onward is untouched by
    training, standardization, and early stopping — it is the genuine out-of-sample
    year the portfolio is judged on.
    """

    in_sample_start: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_SPLIT_IN_SAMPLE_START", "2023-10-02")
    )
    in_sample_end: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_SPLIT_IN_SAMPLE_END", "2024-12-31")
    )
    # first train_fraction of in-sample days train, the remainder validate
    train_fraction: float = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_SPLIT_TRAIN_FRACTION", 0.8, float)
    )
    test_start: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_SPLIT_TEST_START", "2025-01-01")
    )
    test_end: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_SPLIT_TEST_END", "2025-12-31")
    )


@dataclass(frozen=True)
class WindowConfig:
    lookback: int = field(  # T: historical trading days per sample
        default_factory=lambda: _env("TYCHE_PORTFOLIO_WINDOW_LOOKBACK", 30, int)
    )
    holding: int = field(  # H: forward return horizon (also rebalance step)
        default_factory=lambda: _env("TYCHE_PORTFOLIO_WINDOW_HOLDING", 5, int)
    )
    embargo: int = field(  # >= H trading days between splits
        default_factory=lambda: _env("TYCHE_PORTFOLIO_WINDOW_EMBARGO", 5, int)
    )


@dataclass(frozen=True)
class IntradayConfig:
    resample: str = field(  # 5min | 15min | 30min
        default_factory=lambda: _env("TYCHE_PORTFOLIO_INTRADAY_RESAMPLE", "15min")
    )
    session_start: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_INTRADAY_SESSION_START", "09:30")
    )
    session_end: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_INTRADAY_SESSION_END", "16:00")
    )
    max_bars_per_day: int = field(  # 6.5h / 15min ~ 26 regular bars (+slack)
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_INTRADAY_MAX_BARS_PER_DAY", 27, int
        )
    )


@dataclass(frozen=True)
class DailyFeatureConfig:
    vol_window: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_VOL_WINDOW", 20, int)
    )
    mom_windows: tuple[int, ...] = field(
        default_factory=lambda: _env_int_tuple(
            "TYCHE_PORTFOLIO_DAILY_MOM_WINDOWS", (5, 20)
        )
    )
    rsi_window: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_RSI_WINDOW", 14, int)
    )
    atr_window: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_ATR_WINDOW", 14, int)
    )
    zscore_window: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_ZSCORE_WINDOW", 20, int)
    )


@dataclass(frozen=True)
class NewsFeatureConfig:
    """Portfolio-side news-story clustering before daily aggregation."""

    dedup_enabled: bool = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_NEWS_DEDUP_ENABLED", True, bool)
    )
    dedup_lookback_days: int = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_NEWS_DEDUP_LOOKBACK_DAYS", 30, int
        )
    )
    dedup_similarity_threshold: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_NEWS_DEDUP_SIMILARITY_THRESHOLD", 0.90, float
        )
    )
    # ``auto`` selects CUDA, then MPS, then CPU. This accelerates cosine graph creation.
    dedup_device: str = field(  # auto | cpu | cuda | cuda:N | mps
        default_factory=lambda: _env("TYCHE_PORTFOLIO_NEWS_DEDUP_DEVICE", "auto")
    )
    dedup_similarity_batch_size: int = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_NEWS_DEDUP_SIMILARITY_BATCH_SIZE", 1_024, int
        )
    )


@dataclass(frozen=True)
class ModelConfig:
    conv_channels: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_CONV_CHANNELS", 32, int)
    )
    kernel_size: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_KERNEL_SIZE", 3, int)
    )
    dropout: float = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_DROPOUT", 0.2, float)
    )
    hidden_dim: int = field(  # LSTM hidden size / fused day-embedding size
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_HIDDEN_DIM", 64, int)
    )
    cov_rank: int = field(  # low-rank factor width for Sigma = LL^T + diag(d)
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_COV_RANK", 2, int)
    )
    cov_eps: float = field(  # softplus floor + Cholesky jitter
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_COV_EPS", 1e-4, float)
    )
    sequence_encoder: str = field(  # lstm | attention
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_MODEL_SEQUENCE_ENCODER", "lstm"
        )
    )


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_EPOCHS", 40, int)
    )
    batch_size: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_BATCH_SIZE", 32, int)
    )
    lr: float = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_LR", 1e-3, float)
    )
    weight_decay: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_TRAIN_WEIGHT_DECAY", 1e-5, float
        )
    )
    target_distribution: str = field(  # gaussian | student_t
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_TRAIN_TARGET_DISTRIBUTION", "student_t"
        )
    )
    student_t_df: float = field(  # degrees of freedom; must be > 2 for finite covariance
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_TRAIN_STUDENT_T_DF", 5.0, float
        )
    )
    # Stochastic forward passes at prediction time. Dropout stays enabled only for
    # these passes, yielding epistemic covariance from mean-prediction disagreement.
    mc_dropout_samples: int = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_TRAIN_MC_DROPOUT_SAMPLES", 50, int
        )
    )
    grad_clip: float = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_GRAD_CLIP", 1.0, float)
    )
    patience: int = field(  # early-stopping on val NLL
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_PATIENCE", 8, int)
    )
    seed: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_SEED", 7, int)
    )
    device: str = field(  # auto | cpu | cuda | mps
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_DEVICE", "auto")
    )


@dataclass(frozen=True)
class PortfolioConfig:
    """Black-Litterman, risk-based allocators, mean-variance optimization, backtest."""

    bl_tau: float = field(  # prior-covariance scaling in BL
        default_factory=lambda: _env("TYCHE_PORTFOLIO_ALLOC_BL_TAU", 0.05, float)
    )
    bl_risk_aversion: float = field(  # delta, implied-equilibrium-return scaling
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_BL_RISK_AVERSION", 2.5, float
        )
    )
    cov_shrinkage: float = field(  # blend predicted vs reference covariance
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_COV_SHRINKAGE", 0.3, float
        )
    )
    max_weight: float = field(  # per-asset cap (constrained MVO only)
        default_factory=lambda: _env("TYCHE_PORTFOLIO_ALLOC_MAX_WEIGHT", 0.40, float)
    )
    turnover_penalty: float = field(  # optional L1 turnover term in the MVO objective
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_TURNOVER_PENALTY", 0.0, float
        )
    )
    transaction_cost_bps: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_TRANSACTION_COST_BPS", 10.0, float
        )
    )
    slippage_bps: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_SLIPPAGE_BPS", 5.0, float
        )
    )
    # Predicted mu/Sigma are log-return moments (the training target is a log return);
    # BL, the optimizers, and the cost model all assume arithmetic returns. When true,
    # the exact lognormal moment transform is applied before allocation.
    convert_to_simple_returns: bool = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_CONVERT_TO_SIMPLE_RETURNS", True, bool
        )
    )

    # "round_trip" applies R' = (R(1-x) - 2x) / (1+x) per holding period with x scaled
    # by realized turnover; "linear" charges the older turnover * cost_rate.
    cost_model: str = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_COST_MODEL", "round_trip"
        )
    )

    # Risk parity (equal risk contribution), solved by cyclical coordinate descent.
    risk_parity_max_iter: int = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_RISK_PARITY_MAX_ITER", 500, int
        )
    )
    risk_parity_tol: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_RISK_PARITY_TOL", 1e-10, float
        )
    )
    # Hierarchical risk parity: scipy linkage method on the correlation distance.
    hrp_linkage: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_ALLOC_HRP_LINKAGE", "single")
    )


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    split: SplitConfig = field(default_factory=SplitConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    intraday: IntradayConfig = field(default_factory=IntradayConfig)
    daily: DailyFeatureConfig = field(default_factory=DailyFeatureConfig)
    news: NewsFeatureConfig = field(default_factory=NewsFeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    @property
    def artifacts_dir(self) -> Path:
        """Benchmark output directory for this run, split per target distribution
        (e.g. ``benchmark/student_t``, ``benchmark/gaussian``)."""
        return self.paths.artifacts / self.train.target_distribution


def default_config() -> Config:
    """Build a ``Config`` from the current environment (``.env`` + os.environ)."""
    return Config()
