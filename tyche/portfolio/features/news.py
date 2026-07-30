"""News feature branch — centroid-representative story sentiment.

For each ``(stock, trading-day)`` this looks back over a configurable one-month
selection window, clusters near-duplicate article summaries with local text embeddings,
and treats each cluster as one news story. Each cluster is represented by the article
closest to the cluster centroid. Daily features then use only those representative
article scores: the mean representative sentiment and the log number of unique story
clusters in the trailing selection window.

Every feature is built from articles published at or before the trading day it lands
on. Days with no news are zeros — never forward-filled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from tyche.common.logging import get_logger
from tyche.news.service.embedder import embed_texts
from tyche.portfolio.config import Config

log = get_logger(__name__)

NEWS_FEATURES: list[str] = [
    "mean_sent",
    "log_n_articles",
]


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x


def _cluster_labels(embeddings: np.ndarray, similarity_threshold: float) -> np.ndarray:
    if len(embeddings) == 1:
        return np.zeros(1, dtype=int)
    distance_threshold = 1.0 - float(similarity_threshold)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return clustering.fit_predict(embeddings)


def _centroid_representatives(embeddings: np.ndarray, labels: np.ndarray) -> list[int]:
    representatives: list[int] = []
    for label in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == label)
        cluster = embeddings[members]
        centroid = _unit(cluster.mean(axis=0))
        scores = cluster @ centroid
        representatives.append(int(members[int(np.argmax(scores))]))
    return representatives


def _representative_indices(
    group: pd.DataFrame,
    embedding_by_text: dict[str, np.ndarray],
    similarity_threshold: float,
) -> list[int]:
    summary = group["summary_text"].fillna("").astype(str).str.strip()
    embed_rows = [idx for idx, text in summary.items() if text in embedding_by_text]
    keep_rows = [idx for idx, text in summary.items() if text not in embedding_by_text]
    if len(embed_rows) <= 1:
        return [*keep_rows, *embed_rows]

    embeddings = np.vstack(
        [embedding_by_text[str(summary.loc[idx])] for idx in embed_rows]
    )
    labels = _cluster_labels(embeddings, similarity_threshold)
    reps = _centroid_representatives(embeddings, labels)
    return [*keep_rows, *[embed_rows[i] for i in reps]]


def _load_embedding_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        cache = np.load(path, allow_pickle=True)
        texts = [str(t) for t in cache["texts"].tolist()]
        embeddings = cache["embeddings"].astype(np.float32)
    except Exception as exc:
        log.warning("could not load news embedding cache %s: %s", path, exc)
        return {}
    if len(texts) != len(embeddings):
        log.warning(
            "ignoring invalid news embedding cache %s: %d texts for %d embeddings",
            path,
            len(texts),
            len(embeddings),
        )
        return {}
    return {text: embeddings[i] for i, text in enumerate(texts)}


def _save_embedding_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    texts = list(cache)
    embeddings = np.vstack([cache[text] for text in texts]).astype(np.float32)
    np.savez_compressed(
        path,
        texts=np.asarray(texts, dtype=object),
        embeddings=embeddings,
    )


def _embedding_lookup(df: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    text = df["summary_text"].fillna("").astype(str).str.strip()
    unique_texts = list(dict.fromkeys(t for t in text if t))
    if not unique_texts:
        return {}

    cache_path = cfg.paths.news_embedding_cache
    cache = _load_embedding_cache(Path(cache_path))
    missing = [text for text in unique_texts if text not in cache]
    log.info(
        "portfolio news feature extraction found %d unique summaries from %d "
        "article rows | cached=%d missing=%d",
        len(unique_texts),
        len(df),
        len(unique_texts) - len(missing),
        len(missing),
    )
    if missing:
        log.info("embedding %d missing portfolio news summaries", len(missing))
        embeddings = embed_texts(missing)
        for i, text in enumerate(missing):
            cache[text] = embeddings[i].astype(np.float32)
        _save_embedding_cache(Path(cache_path), cache)
        log.info("saved news embedding cache to %s", cache_path)
    return {text: cache[text] for text in unique_texts if text in cache}


def _representative_window_features(
    window: pd.DataFrame,
    embedding_by_text: dict[str, np.ndarray],
    cfg: Config,
) -> tuple[float, int]:
    keep = _representative_indices(
        window,
        embedding_by_text,
        cfg.news.dedup_similarity_threshold,
    )
    scores = window.loc[keep, "sentiment_final"].astype(float)
    return float(scores.mean()), int(len(scores))


def _aggregate_selection_windows(
    df: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    cfg: Config,
) -> pd.DataFrame:
    """Aggregate representative stories from the trailing selection window."""
    if df.empty:
        return pd.DataFrame(columns=["asset", "date", "mean_sent", "n_articles"])

    if not cfg.news.dedup_enabled:
        return _aggregate(df)

    embedding_by_text = _embedding_lookup(df, cfg)
    lookback = pd.Timedelta(days=int(cfg.news.dedup_lookback_days))
    rows: list[dict] = []
    total_representatives = 0
    grouped = list(df.groupby("asset", sort=True))
    total_windows = len(grouped) * len(trading_days)

    with tqdm(
        total=total_windows,
        desc="news feature windows",
        unit="window",
    ) as pbar:
        for asset, asset_df in grouped:
            asset_df = asset_df.sort_values(["date", "ts"])
            for date in trading_days:
                pbar.update(1)
                pbar.set_postfix_str(f"asset={asset}")
                start = date - lookback
                window = asset_df[
                    (asset_df["date"] > start) & (asset_df["date"] <= date)
                ]
                if window.empty:
                    continue
                mean_sent, n_articles = _representative_window_features(
                    window,
                    embedding_by_text,
                    cfg,
                )
                total_representatives += n_articles
                rows.append(
                    {
                        "asset": asset,
                        "date": date,
                        "mean_sent": mean_sent,
                        "n_articles": n_articles,
                    }
                )

    log.info(
        "news deduplication used %d centroid representatives across %d "
        "asset-day selection windows from %d articles (lookback=%d days)",
        total_representatives,
        len(rows),
        len(df),
        cfg.news.dedup_lookback_days,
    )
    return pd.DataFrame(rows, columns=["asset", "date", "mean_sent", "n_articles"])


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (asset, date) summarizing representative story articles."""
    grouped = df.groupby(["asset", "date"])
    agg = grouped.agg(
        mean_sent=("sentiment_final", "mean"),
        n_articles=("sentiment_final", "size"),
    ).reset_index()
    return agg


def build_news_features(
    news: pd.DataFrame, trading_days: pd.DatetimeIndex, cfg: Config
) -> pd.DataFrame:
    """Return a dense long frame ``[asset, date, *NEWS_FEATURES]`` covering every
    (asset, trading_day) — including no-news days.

    News published on calendar day ``d`` is attributed to trading day ``d`` (known by
    that session's close); news on a non-trading day rolls forward to the next session
    so nothing is dropped, and nothing leaks backward."""
    df = news.copy()
    if "summary_text" not in df.columns:
        df["summary_text"] = ""
    cal_day = pd.DatetimeIndex(df["ts"].dt.normalize())
    # Snap each article to the first trading day >= its calendar day.
    pos = trading_days.searchsorted(cal_day, side="left")
    valid = pos < len(trading_days)
    df = df[valid].copy()
    df["date"] = trading_days[pos[valid]]

    # Dense grid over (assets seen in news) x trading_days, so no-news days exist.
    assets = sorted(df["asset"].unique())
    grid = pd.MultiIndex.from_product(
        [assets, trading_days], names=["asset", "date"]
    ).to_frame(index=False)
    agg = _aggregate_selection_windows(df, trading_days, cfg)
    out = grid.merge(agg, on=["asset", "date"], how="left")

    for col in ("mean_sent", "n_articles"):
        out[col] = out[col].fillna(0.0)

    # Story counts are heavily right-skewed, so they enter on a log scale.
    out["log_n_articles"] = np.log1p(out["n_articles"])

    return out[["asset", "date", *NEWS_FEATURES]]
