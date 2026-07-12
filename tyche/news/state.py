"""LangGraph state schema — the tables that flow through the 6-agent DAG.

Each agent node reads the frame(s) produced upstream and writes its own key; the
graph is linear (a DAG, not a loop), so no reducer is needed — later writes to a
key simply replace it.
"""

from __future__ import annotations

from typing import Optional, TypedDict

import pandas as pd


class PipelineState(TypedDict, total=False):
    input_path: Optional[str]
    ingested: pd.DataFrame
    spans: pd.DataFrame
    scored: pd.DataFrame
    aggregated: pd.DataFrame
    neutralized: pd.DataFrame
    audit: dict
