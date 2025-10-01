# pip install langchain-mcp-adapters
import os
from typing import List
from typing_extensions import TypedDict
from typing import Annotated
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain_openai import ChatOpenAI
import asyncio

# 初始化多服务器客户端
client = MultiServerMCPClient(
    {
        # "math": {
        #     "command": "python",
        #     "args": ["math_mcp_server.py"],  # 确保这个文件存在且可运行
        #     "transport": "stdio",
        # },
        "logistics": {
            "url": "http://localhost:8000/mcp",  # 物流MCP服务地址
            "transport": "streamable_http",
        }
    }
)

async def create_graph(logistics_session):
    # 使用 OpenAI 模型
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    # 加载工具
    logistics_tools = await load_mcp_tools(logistics_session)
    tools = logistics_tools

    # 绑定工具到 LLM
    llm_with_tool = llm.bind_tools(tools)

    # 可选：从 MCP 加载 system prompt
    try:
        system_prompt_msg = await load_mcp_prompt(logistics_session, "system_prompt")
        system_prompt = system_prompt_msg[0].content
    except Exception as e:
        print("未加载到 system prompt，使用默认提示。错误:", e)
        system_prompt = "你是一个智能助手，可以调用工具来回答用户问题。"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    chat_llm = prompt_template | llm_with_tool

    # 状态定义
    class State(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]

    # 节点函数
    def chat_node(state: State) -> State:
        response = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    tool_node = ToolNode(tools=tools)

    # 构建图
    graph_builder = StateGraph(State)
    graph_builder.add_node("chat_node", chat_node)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.add_edge(START, "chat_node")
    graph_builder.add_conditional_edges(
        "chat_node",
        tools_condition,
        {"tools": "tool_node", "__end__": END}
    )
    graph_builder.add_edge("tool_node", "chat_node")
    
    graph = graph_builder.compile(checkpointer=MemorySaver())
    return graph


async def main():
    config = {"configurable": {"thread_id": "logistics_thread_001"}}
    
    async with client.session("logistics") as logistics_session:

        print("MCP 客户端已连接：Logistics 服务")
        agent = await create_graph(logistics_session)

        print("\n欢迎使用智能物流客服！你可以询问：")
        print("包裹状态、运费、送达时间等")
        print("输入 'quit' 退出\n")

        while True:
            try:
                user_input = input("User: ").strip()
                if user_input.lower() in ["quit", "exit", "退出"]:
                    print("再见！")
                    break

                # 调用代理
                response = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config
                )
                ai_message = response["messages"][-1].content
                print(f"AI: {ai_message}\n")

            except KeyboardInterrupt:
                print("\n已退出。")
                break
            except Exception as e:
                print(f"出错: {e}")

if __name__ == "__main__":
    asyncio.run(main())



# User: 我的包裹 LGT123456 到哪了？
# AI: 包裹 LGT123456 的当前状态是：已发货，在途中。

# User: 从北京到上海寄一个5公斤的包裹要多少钱？距离大约是1200公里。
# AI: 从北京到上海寄一个5公斤的包裹，距离大约1200公里，运费估算为27.0元。还有其他需要帮忙的吗？

# User: 那大概多久能到？
# AI: 基于距离1200公里的估算，预计送达时间约1天6小时（约34小时）。但实际时效可能受快递公司、起运地/目的地、天气、节假日等因素影响，可能有延迟。
