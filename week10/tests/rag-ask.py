import os
import asyncio
import logging
import dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.memory import InMemorySaver

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """检索技术知识库，返回序列化内容与原始文档"""
    logging.info("retrieve_context.input: %s", query)
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    print(f"检索到的文档: {retrieved_docs}")
    return serialized, retrieved_docs

prompt = (
    """
        你是一个严谨的客服问答助手。你的回答必须只依据“参考资料”的 Content 字段。
        如需检索，先调用 retrieve_context，并仅使用返回的 Content

        格式处理：
        - 保持自然表述；可轻度改写但不得改变含义
    """
)

agent = create_agent(
    model=ChatTongyi(model="qwen-turbo"),
    tools=[retrieve_context],
    checkpointer=InMemorySaver(),
    system_prompt=prompt,
)

async def main():
    query = "应届生能听懂吗？"
    print(f"查询: {query}")

    serialized, retrieved_docs = retrieve_context.func(query)

    try:
        meta_list = [doc.metadata for doc in (retrieved_docs or [])]
    except Exception as e:
        print("读取文档元数据异常:", e)
        meta_list = []
    print("工具输出(元数据):", meta_list)

    source = (retrieved_docs[0].metadata.get("source") if retrieved_docs else None)
    user_content = (
        f"参考资料：\n{serialized}\n\n"
        f"问题：{query}\n\n"
    )
    messages = {"messages": [{"role": "user", "content": user_content}]}
    config = {"configurable": {"thread_id": "1"}}

    async for step in agent.astream(messages, config=config, stream_mode="values"):
        msgs = step.get("messages", [])
        if msgs:
            m = msgs[-1]
            c = getattr(m, "content", None)
            if isinstance(c, str) and c.strip():
                print(c, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
