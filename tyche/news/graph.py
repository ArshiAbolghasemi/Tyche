"""The agent DAG, wired as a LangGraph ``StateGraph``.

    ingest → summarizer → scorer → neutralizer → auditor → END

Each node is a thin wrapper that calls one agent and returns the state key it
produces. The Summarizer compresses each article with ``facebook/bart-large-cnn``
and the Scorer runs FinBERT on that single summary — so there is one score per
(article, ticker) and no span-aggregation step.
"""

from __future__ import annotations

from tyche.news.agents import (
    auditor,
    ingest,
    neutralizer,
    scorer,
    summarizer,
)
from tyche.news.state import PipelineState


def _ingest(state: PipelineState) -> dict:
    return {"ingested": ingest.ingest(state.get("input_path"))}


def _summarize(state: PipelineState) -> dict:
    return {"summarized": summarizer.summarize(state["ingested"])}


def _score(state: PipelineState) -> dict:
    return {"scored": scorer.score(state["summarized"])}


def _neutralize(state: PipelineState) -> dict:
    return {"neutralized": neutralizer.neutralize(state["scored"])}


def _audit(state: PipelineState) -> dict:
    return {"audit": auditor.audit_d(state["neutralized"])}


def build_graph():
    """Compile the linear DAG. Returns a runnable graph."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(PipelineState)
    graph.add_node("ingest", _ingest)
    graph.add_node("summarizer", _summarize)
    graph.add_node("scorer", _score)
    graph.add_node("neutralizer", _neutralize)
    graph.add_node("auditor", _audit)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "summarizer")
    graph.add_edge("summarizer", "scorer")
    graph.add_edge("scorer", "neutralizer")
    graph.add_edge("neutralizer", "auditor")
    graph.add_edge("auditor", END)
    return graph.compile()
