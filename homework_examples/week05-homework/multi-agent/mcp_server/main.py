import json
from fastmcp import FastMCP
from duckduckgo_search import DDGS
from .prompts import PROMPTS


mcp = FastMCP("Writer Agent Tools")


@mcp.tool
def search(topic: str, max_results: int = 5) -> str:
    """根据主题进行网络搜索，并返回JSON格式的搜索结果。"""
    print(f"MCP Server: 🔍 Executing search for '{topic}'...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(topic, max_results=max_results))
            return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool
def get_prompt(agent_name: str) -> str:
    """根据代理名称获取对应的系统提示词。"""
    print(f"MCP Server: 📄 Providing prompt for '{agent_name}'...")
    return PROMPTS.get(agent_name, "Error: Prompt not found.")


def run():
    """运行 FastMCP HTTP 服务。"""
    print("🚀 MCP Server (HTTP) is running at http://localhost:8000/mcp")
    # 使用 streamable-http 传输方式
    mcp.run(transport="streamable-http", port=8000)


if __name__ == "__main__":
    run()