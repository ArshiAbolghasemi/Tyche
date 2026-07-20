"""The agent DAG, wired as a LangGraph ``StateGraph``.

    ingest → [summarizer] → deduplicator → scorer → neutralizer → auditor → END

Each node is a thin wrapper that calls one agent and returns the state key it
produces. The Summarizer compresses each article with ``facebook/bart-large-cnn``;
the Deduplicator embeds those summaries with ``BAAI/bge-m3`` and collapses monthly
near-duplicates to one cluster representative; the Scorer then extracts sentiment for
each representative with an Azure OpenAI model — one score per (article, ticker) row,
shared across a cluster, with no span-aggregation step.

The summarizer node is conditional: some sources (e.g. zanista) ship a pre-computed
``summary`` for a subset of rows, passed through by ``ingest`` as ``summary_text``. If
every ingested row already has one, the summarizer node — and its BART model load — is
skipped entirely and ``ingested`` flows straight to the deduplicator as ``summarized``.
Otherwise the summarizer node runs, and it internally reuses any pre-existing
``summary_text`` per row and only generates the ones still missing.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from tyche.news.agents import (
    auditor,
    deduplicator,
    ingest,
    neutralizer,
    scorer,
    summarizer,
)
from tyche.news.records import Summary
from tyche.news.state import PipelineState


def _ingest(state: PipelineState) -> dict:
    return {"ingested": ingest.ingest(state.get("input_path"))}


def _needs_summarizer(state: PipelineState) -> str:
    """Route to the summarizer node unless every ingested row already carries a
    non-empty pre-existing summary (in which case there's nothing left to generate)."""
    df = state["ingested"]
    if df.empty or Summary.text not in df.columns:
        return "summarizer"
    if df[Summary.text].fillna("").eq("").any():
        return "summarizer"
    return "deduplicator"


def _summarize(state: PipelineState) -> dict:
    return {"summarized": summarizer.summarize(state["ingested"])}


def _skip_summarizer(state: PipelineState) -> dict:
    return {"summarized": state["ingested"]}


def _deduplicate(state: PipelineState) -> dict:
    return {"deduplicated": deduplicator.deduplicate(state["summarized"])}


def _score(state: PipelineState) -> dict:
    return {"scored": scorer.score(state["deduplicated"])}


def _neutralize(state: PipelineState) -> dict:
    return {"neutralized": neutralizer.neutralize(state["scored"])}


def _audit(state: PipelineState) -> dict:
    return {"audit": auditor.audit_d(state["neutralized"])}


def build_graph():
    """Compile the DAG. Returns a runnable graph."""

    graph = StateGraph(PipelineState)
    graph.add_node("ingest", _ingest)
    graph.add_node("summarizer", _summarize)
    graph.add_node("skip_summarizer", _skip_summarizer)
    graph.add_node("deduplicator", _deduplicate)
    graph.add_node("scorer", _score)
    graph.add_node("neutralizer", _neutralize)
    graph.add_node("auditor", _audit)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges(
        "ingest",
        _needs_summarizer,
        {"summarizer": "summarizer", "deduplicator": "skip_summarizer"},
    )
    graph.add_edge("summarizer", "deduplicator")
    graph.add_edge("skip_summarizer", "deduplicator")
    graph.add_edge("deduplicator", "scorer")
    graph.add_edge("scorer", "neutralizer")
    graph.add_edge("neutralizer", "auditor")
    graph.add_edge("auditor", END)
    return graph.compile()
