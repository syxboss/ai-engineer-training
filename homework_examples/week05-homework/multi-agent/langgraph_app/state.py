from typing import TypedDict, List, Any
from fastmcp import Client


class AgentState(TypedDict):
    """定义图的状态"""
    topic: str
    style: str
    length: int
    research_report: str
    draft: str
    review_suggestions: str
    final_article: str
    log: List[str]
    mcp_client: Any
