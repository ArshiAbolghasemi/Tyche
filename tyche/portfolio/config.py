"""Central configuration for the multimodal portfolio pipeline.

Every stage reads its settings from the frozen dataclasses here, so paths, split
dates, window sizes, and hyperparameters live in exactly one place. Defaults target
the data actually on disk (prices cover 2023-10-02 -> 2024-12-31, 5 assets); set a
real ``test`` range once out-of-sample prices exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    news_sentiment: Path = REPO_ROOT / "data/output/news_sentiment.parquet"
    daily_ohlcv: Path = REPO_ROOT / "data/eodhd/daily_ohclv.parquet"
    intraday_ohlcv: Path = REPO_ROOT / "data/eodhd/intraday_ohclv.parquet"
    artifacts: Path = REPO_ROOT / "tyche/portfolio/artifacts"
    news_embedding_cache: Path = (
        REPO_ROOT / "tyche/portfolio/artifacts/news_embedding_cache.npz"
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

    in_sample_start: str = "2023-10-02"
    in_sample_end: str = "2024-12-31"
    train_fraction: float = 0.8  # first 80% of in-sample days train, last 20% validate
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"


@dataclass(frozen=True)
class WindowConfig:
    lookback: int = 30  # T: historical trading days per sample
    holding: int = 5  # H: forward return horizon (also rebalance step)
    embargo: int = 5  # >= H trading days between splits


@dataclass(frozen=True)
class IntradayConfig:
    resample: str = "15min"  # 5min | 15min | 30min
    session_start: str = "09:30"
    session_end: str = "16:00"
    max_bars_per_day: int = 27  # 6.5h / 15min ~ 26 regular bars (+slack)


@dataclass(frozen=True)
class DailyFeatureConfig:
    vol_window: int = 20
    mom_windows: tuple[int, ...] = (5, 20)
    rsi_window: int = 14
    atr_window: int = 14
    zscore_window: int = 20


@dataclass(frozen=True)
class NewsFeatureConfig:
    """Portfolio-side news-story clustering before daily aggregation."""

    dedup_enabled: bool = True
    dedup_lookback_days: int = 30
    dedup_similarity_threshold: float = 0.90
    # ``auto`` selects CUDA, then MPS, then CPU. This accelerates cosine graph creation.
    dedup_device: str = "auto"  # auto | cpu | cuda | cuda:N | mps
    dedup_similarity_batch_size: int = 1_024


@dataclass(frozen=True)
class ModelConfig:
    conv_channels: int = 32
    kernel_size: int = 3
    dropout: float = 0.2
    hidden_dim: int = 64  # LSTM hidden size / fused day-embedding size
    cov_rank: int = 2  # low-rank factor width for Sigma = LL^T + diag(d)
    cov_eps: float = 1e-4  # softplus floor + Cholesky jitter
    sequence_encoder: str = "lstm"  # lstm | attention


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    target_distribution: str = "gaussian"  # gaussian | student_t
    student_t_df: float = 5.0  # degrees of freedom; must be > 2 for finite covariance
    huber_lambda: float = 1.0  # lambda_1
    cov_reg_lambda: float = 1e-3  # lambda_2
    grad_clip: float = 1.0
    patience: int = 8  # early-stopping on val NLL
    seed: int = 7
    device: str = "auto"  # auto | cpu | cuda | mps


@dataclass(frozen=True)
class PortfolioConfig:
    """Black-Litterman, risk-based allocators, mean-variance optimization, backtest."""

    bl_tau: float = 0.05  # prior-covariance scaling in BL
    bl_risk_aversion: float = 2.5  # delta, implied-equilibrium-return scaling
    cov_shrinkage: float = 0.3  # blend predicted vs reference covariance
    max_weight: float = 0.40  # per-asset cap (constrained MVO only)
    turnover_penalty: float = 0.0  # optional L1 turnover term in the MVO objective
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    # Predicted mu/Sigma are log-return moments (the training target is a log return);
    # BL, the optimizers, and the cost model all assume arithmetic returns. When true,
    # the exact lognormal moment transform is applied before allocation.
    convert_to_simple_returns: bool = True

    # "round_trip" applies R' = (R(1-x) - 2x) / (1+x) per holding period with x scaled
    # by realized turnover; "linear" charges the older turnover * cost_rate.
    cost_model: str = "round_trip"

    # Risk parity (equal risk contribution), solved by cyclical coordinate descent.
    risk_parity_max_iter: int = 500
    risk_parity_tol: float = 1e-10
    # Hierarchical risk parity: scipy linkage method on the correlation distance.
    hrp_linkage: str = "single"


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


def default_config() -> Config:
    return Config()
