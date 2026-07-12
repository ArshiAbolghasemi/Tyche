"""The 6-agent DAG, wired as a LangGraph ``StateGraph``.

    ingest → segmentation → scorer → aggregator → neutralizer → auditor → END

Each node is a thin wrapper that calls one agent and returns the state key it
produces. No agent calls an LLM; all intelligence is FinBERT inference plus
deterministic logic.
"""

from __future__ import annotations

from tyche.news.agents import (
    aggregator,
    auditor,
    ingest,
    neutralizer,
    scorer,
    segmentation,
)
from tyche.news.state import PipelineState


def _ingest(state: PipelineState) -> dict:
    return {"ingested": ingest.ingest(state.get("input_path"))}


def _segment(state: PipelineState) -> dict:
    return {"spans": segmentation.segment(state["ingested"])}


def _score(state: PipelineState) -> dict:
    return {"scored": scorer.score(state["spans"])}


def _aggregate(state: PipelineState) -> dict:
    return {"aggregated": aggregator.aggregate(state["scored"], state["ingested"])}


def _neutralize(state: PipelineState) -> dict:
    return {"neutralized": neutralizer.neutralize(state["aggregated"])}


def _audit(state: PipelineState) -> dict:
    return {"audit": auditor.audit_d(state["neutralized"])}


def build_graph():
    """Compile the linear DAG. Returns a runnable graph."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(PipelineState)
    graph.add_node("ingest", _ingest)
    graph.add_node("segmentation", _segment)
    graph.add_node("scorer", _score)
    graph.add_node("aggregator", _aggregate)
    graph.add_node("neutralizer", _neutralize)
    graph.add_node("auditor", _audit)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "segmentation")
    graph.add_edge("segmentation", "scorer")
    graph.add_edge("scorer", "aggregator")
    graph.add_edge("aggregator", "neutralizer")
    graph.add_edge("neutralizer", "auditor")
    graph.add_edge("auditor", END)
    return graph.compile()
