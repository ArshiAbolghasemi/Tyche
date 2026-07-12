"""Agent 4 — Aggregator. Span scores → one score per (article, ticker).

Span weight combines conviction ``(1 − p_neu)`` with an inverted-pyramid position
decay ``exp(−λ · pos/n_spans)`` (the lead sentence carries the verdict), discounted
when a span was flagged irrelevant.
"""

from __future__ import annotations

import math

import pandas as pd

from tyche.common.config import settings
from tyche.common.logging import get_logger
from tyche.news.records import Aggregate, Article, Score, Span

log = get_logger(__name__)


def _aggregate_group(group: pd.DataFrame) -> pd.Series:
    lam = float(settings.aggregation.position_lambda)
    discount = float(settings.aggregation.irrelevant_discount)
    eps = float(settings.aggregation.weight_epsilon)
    n = len(group)

    weights = []
    for _, span in group.iterrows():
        pos_w = math.exp(-lam * (span[Span.position_index] / n)) if n else 1.0
        w = (1.0 - span[Score.p_neu]) * pos_w
        if not span[Span.relevant]:
            w *= discount
        weights.append(w)
    total = sum(weights)

    if total < eps:  # low-conviction: fall back to simple means
        raw = float(group[Score.span_score].mean())
        p_pos, p_neg, p_neu = (
            float(group[c].mean()) for c in (Score.p_pos, Score.p_neg, Score.p_neu)
        )
    else:
        raw = sum(w * s for w, s in zip(weights, group[Score.span_score])) / total
        p_pos = sum(w * s for w, s in zip(weights, group[Score.p_pos])) / total
        p_neg = sum(w * s for w, s in zip(weights, group[Score.p_neg])) / total
        p_neu = sum(w * s for w, s in zip(weights, group[Score.p_neu])) / total

    norm = p_pos + p_neg + p_neu
    if norm > 0:
        p_pos, p_neg, p_neu = p_pos / norm, p_neg / norm, p_neu / norm

    return pd.Series(
        {
            Aggregate.p_pos: p_pos,
            Aggregate.p_neg: p_neg,
            Aggregate.p_neu: p_neu,
            Aggregate.raw_score: raw,
            Aggregate.n_spans: n,
            Aggregate.n_relevant_spans: int(group[Span.relevant].sum()),
            Score.model_revision: group[Score.model_revision].iloc[0],
        }
    )


def aggregate(scored: pd.DataFrame) -> pd.DataFrame:
    grouped = scored.groupby([Article.id, Article.ticker], sort=False)
    out = grouped.apply(_aggregate_group, include_groups=False).reset_index()
    log.info("aggregated to %d (article, ticker) rows", len(out))
    return out
