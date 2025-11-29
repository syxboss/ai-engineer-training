from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # 用于中断恢复
from typing import List, Dict, Any, Optional
import json
import os
import dotenv
 
# 1. 定义状态结构（融合RAG流程所需的所有数据）
class RAGState:
    """RAG系统的状态管理类，保存整个流程的中间结果"""
    user_query: str = ""  # 用户原始查询
    intent: Optional[str] = None  # 识别的意图
    retrieval_results: List[Dict] = []  # 检索结果（RAG的检索环节输出）
    tool_calls: List[Dict] = []  # 工具调用记录
    tool_results: List[Dict] = []  # 工具返回结果
    thinking_process: List[str] = []  # 思考过程（下周计划实现）
    streaming_output: List[str] = []  # 流式输出内容
    checkpoint: Optional[Dict] = None  # 检查点数据（用于中断恢复）
 
# 2. 初始化图和检查点（支持中断恢复的基础）
memory = MemorySaver()  # 存储检查点的内存管理器
workflow = StateGraph(RAGState)
dotenv.load_dotenv()
 
# 3. 定义核心节点函数
def intent_recognition_node(state: RAGState) -> Dict[str, Any]:
    """意图识别节点：判断用户查询是否需要检索或工具调用"""
    from langchain_community.chat_models import ChatTongyi
    llm = ChatTongyi(model="qwen-turbo")
    prompt = f"""
    分析用户查询的意图类型，返回以下之一：
    - "retrieval": 需要检索知识库（如咨询课程、售前、售后问题）
    - "tool": 需要调用工具（如查询订单、付款、进展等情况）
    - "direct": 可直接回答（如需要人工解答）
    
    用户查询：{state.user_query}
    """
    response = llm.invoke(prompt)
    intent = str(getattr(response, "content", response)).strip().lower()
    print(f"识别意图: {intent}")
    return {"intent": intent}
 
# 4. 知识检索节点（当前版本固定参数）
def retrieval_node(state: RAGState) -> Dict[str, Any]:
    """RAG检索节点：从知识库获取相关文档"""
    if state.intent != "retrieval":
        return {"retrieval_results": []}
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import DashScopeEmbeddings
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v4", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    vector_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    results = vector_db.similarity_search(
        query=state.user_query,
        k=3
    )
    formatted_results = [
        {"content": doc.page_content, "score": doc.metadata.get("score", 0)}
        for doc in results
    ]
    return {"retrieval_results": formatted_results}

def _test_intent(query: str) -> Dict[str, Any]:
    state = RAGState()
    state.user_query = query
    return intent_recognition_node(state)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]).strip() or input("输入查询：").strip()
    res = _test_intent(q)
    val = res.get("intent")
    print(val if val is not None else res)


