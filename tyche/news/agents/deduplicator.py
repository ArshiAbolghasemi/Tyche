"""Agent 4 — Deduplicator. Collapse near-duplicate summaries before sentiment.

Financial news is heavily syndicated: the same story is reprinted across outlets with
minor edits. Scoring every reprint wastes (paid) LLM sentiment calls and over-weights
whichever story got the most reprints. This agent deduplicates **one calendar month at
a time** (per the config ``window``) with an embedding + clustering pass:

1. Bucket rows by calendar month of ``valid_time``.
2. Within each month, embed every *unique* summary with ``BAAI/bge-m3``.
3. Agglomeratively cluster the embeddings by cosine distance (``distance_threshold``);
   each cluster is one underlying story.
4. Compute the cluster **centroid** (mean of the L2-normalized embeddings) and pick the
   member **closest to the centroid** as the cluster representative.
5. Every row inherits its cluster representative's summary (``representative_summary``),
   so the downstream Scorer runs **once per cluster** and shares the score across the
   cluster's members.

No rows are dropped — dedup collapses *scoring work*, not the output table, so the
Neutralizer still sees every (article, ticker) row for its rolling-window statistics.
The month bucketing keeps clustering O(month²) instead of O(corpus²) and matches the
"deduplicate every one month" requirement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from tyche.common.config import settings
from tyche.common.logging import get_logger
from tyche.news.service import embedder
from tyche.news.records import Article, Dedup, Summary

log = get_logger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _cluster_labels(vectors: np.ndarray) -> np.ndarray:
    """Cluster ``vectors`` by cosine distance; return an integer label per row.

    Fewer than two points can't be clustered — they trivially form one cluster."""
    if len(vectors) < 2:
        return np.zeros(len(vectors), dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=float(settings.dedup.distance_threshold),
    )
    return model.fit_predict(vectors)


def _representatives(
    summaries: list[str], vectors: np.ndarray, labels: np.ndarray
) -> dict[str, str]:
    """Map every unique summary to its cluster representative (closest to centroid).

    Centroid is the mean of the L2-normalized member embeddings; the representative is
    the member with the highest cosine similarity to that centroid."""
    unit = _normalize(vectors)
    rep_of_summary: dict[str, str] = {}
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        centroid = unit[members].mean(axis=0)
        sims = unit[members] @ centroid
        rep_summary = summaries[members[int(np.argmax(sims))]]
        for m in members:
            rep_of_summary[summaries[m]] = rep_summary
    return rep_of_summary


def _dedup_month(month_df: pd.DataFrame, month: pd.Period) -> pd.DataFrame:
    """Deduplicate one month's rows in place-ish, returning a labeled copy."""
    out = month_df.copy()
    unique_summaries = out[Summary.text].dropna().unique().tolist()

    if not unique_summaries:
        out[Dedup.cluster_id] = ""
        out[Dedup.representative_text] = out[Summary.text]
        out[Dedup.is_representative] = False
        return out

    vectors = embedder.embed_texts(unique_summaries)
    labels = _cluster_labels(vectors)
    rep_of_summary = _representatives(unique_summaries, vectors, labels)
    # Month-scoped, globally-unique cluster id: "2024-03#7".
    cluster_id_of_summary = {
        summ: f"{month}#{int(labels[i])}" for i, summ in enumerate(unique_summaries)
    }

    out[Dedup.cluster_id] = out[Summary.text].map(cluster_id_of_summary).fillna("")
    out[Dedup.representative_text] = (
        out[Summary.text].map(rep_of_summary).fillna(out[Summary.text])
    )
    out[Dedup.is_representative] = out[Summary.text] == out[Dedup.representative_text]
    log.info(
        "month %s: %d rows, %d unique summaries → %d clusters",
        month,
        len(out),
        len(unique_summaries),
        len(np.unique(labels)),
    )
    return out


def deduplicate(summarized: pd.DataFrame) -> pd.DataFrame:
    """Add dedup columns: ``dedup_month``, ``dedup_cluster_id``, ``is_representative``
    and ``representative_summary`` (the summary the Scorer will actually score).

    Rows are grouped by calendar month and clustered independently within each month.
    Every row keeps its own ``summary_text``; only the *scoring target*
    (``representative_summary``) is deduplicated."""
    if summarized.empty:
        out = summarized.copy()
        out[Dedup.month] = pd.Series(dtype="string")
        out[Dedup.cluster_id] = pd.Series(dtype="string")
        out[Dedup.is_representative] = pd.Series(dtype="bool")
        out[Dedup.representative_text] = pd.Series(dtype="string")
        return out

    df = summarized.copy()
    valid_time = pd.to_datetime(df[Article.valid_time], utc=True)
    # ``to_period`` needs a period-frequency alias ("M" = calendar month), so tz-aware
    # timestamps are localized off first.
    df[Dedup.month] = (
        valid_time.dt.tz_localize(None)
        .dt.to_period(str(settings.dedup.window))
        .astype(str)
    )

    log.info(
        "deduplicating %d rows across %d month buckets (window=%s, cosine threshold=%.3f)",
        len(df),
        df[Dedup.month].nunique(),
        settings.dedup.window,
        float(settings.dedup.distance_threshold),
    )

    try:
        parts = [
            _dedup_month(sub, pd.Period(month))
            for month, sub in df.groupby(Dedup.month, sort=True)
        ]
    finally:
        # All months are done — free the embedder's device memory (CUDA/MPS) for
        # other jobs. Unloading per-month would force a reload every month instead.
        embedder.unload_model()
    out = pd.concat(parts).reindex(df.index)

    n_unique_before = out[Summary.text].nunique()
    n_unique_after = out[Dedup.representative_text].nunique()
    log.info(
        "deduplicated: %d unique summaries → %d cluster representatives "
        "(%.1f%% fewer sentiment calls)",
        n_unique_before,
        n_unique_after,
        100.0 * (1.0 - n_unique_after / max(n_unique_before, 1)),
    )
    return out
