from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.tools import load_mcp_tools
from .state import AgentState
from .nodes import AgentNodes


async def create_graph(mcp_session):
    mcp_tools = await load_mcp_tools(mcp_session)
    nodes = AgentNodes(mcp_tools)
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", nodes.research_node)
    workflow.add_node("writer", nodes.writing_node)
    workflow.add_node("reviewer", nodes.review_node)
    workflow.add_node("polisher", nodes.polishing_node)
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_edge("reviewer", "polisher")
    workflow.add_edge("polisher", END)
    return workflow.compile()