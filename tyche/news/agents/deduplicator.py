"""Agent 4 — Deduplicator. Collapse near-duplicate summaries before sentiment.

Financial news is heavily syndicated: the same story is reprinted across outlets with
minor edits. Scoring every reprint wastes (paid) LLM sentiment calls and over-weights
whichever story got the most reprints. This agent deduplicates **one calendar month at
a time** (per the config ``window``) with an embedding + **online clustering** pass:

1. Bucket rows by calendar month of ``valid_time``.
2. Within each month, embed every *unique* summary with ``BAAI/bge-m3``.
3. Walk the summaries in **publication order** and assign each to the nearest existing
   cluster (cosine distance within ``distance_threshold``) or open a new cluster. The
   comparison is against each cluster's running centroid over the members seen *so far*.
4. The cluster's **seed** — the earliest summary, the one that opened it — is the
   representative. Every later member inherits the seed's summary
   (``representative_summary``), so the downstream Scorer runs **once per cluster** and
   shares the score across the cluster's members.

Why online rather than agglomerative: clustering a whole month at once lets an article
published on the 3rd be grouped — and represented — by an article published on the
28th. The representative's identity, and therefore which rows survive a downstream
``is_representative`` filter, would then depend on up to four weeks of future news. That
is look-ahead for any model keyed on publication date. Online assignment fixes the
direction of inheritance: a member's cluster, representative, and score are always
determined by articles published **at or before** its own timestamp.

Two causal counters ride along for downstream features: ``dedup_cluster_id`` and
``dedup_cluster_size`` — the number of rows in that cluster **up to and including** the
row, i.e. how much coverage the story had accumulated at that moment. Neither peeks
forward.

No rows are dropped — dedup collapses *scoring work*, not the output table, so the
Neutralizer still sees every (article, ticker) row for its rolling-window statistics.
The month bucketing keeps clustering O(month x clusters) instead of O(corpus^2) and
matches the "deduplicate every one month" requirement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tyche.news.config import settings
from tyche.common.logging import get_logger
from tyche.news.service import embedder
from tyche.news.records import Article, Dedup, Summary

log = get_logger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _online_labels(vectors: np.ndarray, threshold: float) -> np.ndarray:
    """Assign each row of ``vectors`` to a cluster using only the rows before it.

    ``vectors`` must already be in publication order and L2-normalized. Row ``i`` joins
    the cluster whose running centroid is closest in cosine distance, provided that
    distance is within ``threshold``; otherwise it opens a new cluster. Centroids are
    maintained as running sums of member vectors, re-normalized on comparison, so a
    cluster's centre reflects exactly the members seen so far and never a future one.
    """
    n = len(vectors)
    if n == 0:
        return np.zeros(0, dtype=int)

    labels = np.empty(n, dtype=int)
    sums: list[np.ndarray] = []  # unnormalized running centroid per cluster

    for i in range(n):
        v = vectors[i]
        if sums:
            centroids = _normalize(np.vstack(sums))
            sims = centroids @ v
            best = int(np.argmax(sims))
            if 1.0 - float(sims[best]) <= threshold:
                labels[i] = best
                sums[best] = sums[best] + v
                continue
        labels[i] = len(sums)
        sums.append(v.copy())

    return labels


def _dedup_month(month_df: pd.DataFrame, month: pd.Period) -> pd.DataFrame:
    """Deduplicate one month's rows online, returning a labeled copy."""
    out = month_df.copy()

    # Unique summaries ordered by their *first* appearance: clustering decisions are
    # made in publication order, and embedding is done once per distinct text.
    first_seen = (
        out.dropna(subset=[Summary.text])
        .groupby(Summary.text, sort=False)[Article.valid_time]
        .min()
        .sort_values()
    )
    unique_summaries = first_seen.index.tolist()

    if not unique_summaries:
        out[Dedup.cluster_id] = ""
        out[Dedup.representative_text] = out[Summary.text]
        out[Dedup.is_representative] = False
        out[Dedup.cluster_size] = 0
        return out

    vectors = _normalize(embedder.embed_texts(unique_summaries))
    labels = _online_labels(vectors, float(settings.dedup.distance_threshold))

    # The seed — first summary of each cluster in publication order — represents it.
    seed_of_label: dict[int, str] = {}
    for summary, label in zip(unique_summaries, labels):
        seed_of_label.setdefault(int(label), summary)

    # Month-scoped, globally-unique cluster id: "2024-03#7".
    cluster_id_of_summary = {
        summ: f"{month}#{int(labels[i])}" for i, summ in enumerate(unique_summaries)
    }
    rep_of_summary = {
        summ: seed_of_label[int(labels[i])] for i, summ in enumerate(unique_summaries)
    }

    out[Dedup.cluster_id] = out[Summary.text].map(cluster_id_of_summary).fillna("")
    out[Dedup.representative_text] = (
        out[Summary.text].map(rep_of_summary).fillna(out[Summary.text])
    )
    out[Dedup.is_representative] = out[Summary.text] == out[Dedup.representative_text]

    # Coverage accumulated so far: rank of each row within its cluster by publication
    # time. Strictly backward-looking, so it is safe as a model feature.
    order = out.sort_values([Dedup.cluster_id, Article.valid_time], kind="mergesort")
    size_so_far = order.groupby(Dedup.cluster_id, sort=False).cumcount() + 1
    out[Dedup.cluster_size] = size_so_far.reindex(out.index).astype(int)

    log.info(
        "month %s: %d rows, %d unique summaries -> %d clusters (online)",
        month,
        len(out),
        len(unique_summaries),
        len(seed_of_label),
    )
    return out


def deduplicate(summarized: pd.DataFrame) -> pd.DataFrame:
    """Add dedup columns: ``dedup_month``, ``dedup_cluster_id``, ``dedup_cluster_size``,
    ``is_representative`` and ``representative_summary`` (the summary the Scorer will
    actually score).

    Rows are grouped by calendar month and clustered **online within each month**, in
    publication order, so no row's cluster or representative depends on an article
    published after it. Every row keeps its own ``summary_text``; only the *scoring
    target* (``representative_summary``) is deduplicated."""
    if summarized.empty:
        out = summarized.copy()
        out[Dedup.month] = pd.Series(dtype="string")
        out[Dedup.cluster_id] = pd.Series(dtype="string")
        out[Dedup.is_representative] = pd.Series(dtype="bool")
        out[Dedup.representative_text] = pd.Series(dtype="string")
        out[Dedup.cluster_size] = pd.Series(dtype="int64")
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
        "deduplicating %d rows across %d month buckets "
        "(window=%s, cosine threshold=%.3f, online)",
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
        "deduplicated: %d unique summaries -> %d cluster representatives "
        "(%.1f%% fewer sentiment calls)",
        n_unique_before,
        n_unique_after,
        100.0 * (1.0 - n_unique_after / max(n_unique_before, 1)),
    )
    return out
