"""News feature branch — centroid-representative story sentiment.

For each ``(stock, trading-day)`` this first clusters near-duplicate article summaries
with local text embeddings. Each cluster is treated as one news story, represented by
the article closest to the cluster centroid. Daily features then use only those
representative article scores: the mean representative sentiment and the log number of
unique story clusters.

Every feature is built from articles published at or before the trading day it lands
on. Days with no news are zeros — never forward-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

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


def _deduplicate_articles(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if df.empty or not cfg.news.dedup_enabled:
        return df

    text = df["summary_text"].fillna("").astype(str).str.strip()
    unique_texts = list(dict.fromkeys(t for t in text if t))
    if not unique_texts:
        return df

    embeddings = embed_texts(unique_texts)
    embedding_by_text = {
        text: embeddings[i].astype(np.float32) for i, text in enumerate(unique_texts)
    }

    keep: list[int] = []
    for _, group in df.groupby(["asset", "date"], sort=False):
        keep.extend(
            _representative_indices(
                group,
                embedding_by_text,
                cfg.news.dedup_similarity_threshold,
            )
        )

    out = df.loc[keep].sort_values(["asset", "date", "ts"]).reset_index(drop=True)
    log.info(
        "news deduplication kept %d centroid representatives from %d articles",
        len(out),
        len(df),
    )
    return out


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
    df = _deduplicate_articles(df, cfg)

    agg = _aggregate(df)

    # Dense grid over (assets seen in news) x trading_days, so no-news days exist.
    assets = sorted(df["asset"].unique())
    grid = pd.MultiIndex.from_product(
        [assets, trading_days], names=["asset", "date"]
    ).to_frame(index=False)
    out = grid.merge(agg, on=["asset", "date"], how="left")

    for col in ("mean_sent", "n_articles"):
        out[col] = out[col].fillna(0.0)

    # Story counts are heavily right-skewed, so they enter on a log scale.
    out["log_n_articles"] = np.log1p(out["n_articles"])

    return out[["asset", "date", *NEWS_FEATURES]]
