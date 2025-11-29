import os
import re
import logging
import dotenv
from typing import Any, Dict, Optional
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# 全局模型实例
llm = ChatTongyi(model="qwen-turbo")

class Route(BaseModel):
    step: Literal["course", "order", "general"] = Field(None)

router_llm = llm.with_structured_output(Route)

# 向量检索缓存
_VECTOR_STORE: Optional[FAISS] = None


# 状态结构
class State(TypedDict, total=False):
    """工作流状态

    字段：
    - query: 清洗后的用户查询
    - intro: 初步引导文案
    - kb_answer: 课程知识库检索答案
    - order_summary: 订单查询自然语言结果
    - direct_answer: 直接答复内容
    - intent: 意图（retrieval | tool | direct）
    """
    query: str
    intro: str
    kb_answer: str
    order_summary: str
    direct_answer: str
    intent: str


def _clean_input(text: str) -> str:
    """输入清洗：去空白、合并空格、去除零宽字符"""
    s = (text or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\u200b", "")  # 去除零宽字符
    return s


def _llm_detect_intent(text: str) -> str:
    """意图识别：检索与订单关键词优先，其余交由 LLM 判断"""
    t_raw = (text or "").strip()
    t = t_raw.lower()
    retrieval_kw = ["课程", "咨询", "问答", "介绍", "知识", "faq", "售前", "售后", "适合", "新手"]
    order_kw = ["订单", "状态", "进度", "支付", "退款", "order", "status"]
    if any(k in t for k in retrieval_kw):
        return "retrieval"
    if any(k in t for k in order_kw):
        return "tool"
    prompt = (
        "请只输出一个词：retrieval、tool 或 direct。\n"
        "当问题涉及课程、适合与否、学习建议、售前售后或知识咨询时，输出 retrieval；\n"
        "当问题涉及订单号、支付、退款、状态、进度时，输出 tool；\n"
        "否则输出 direct。\n\n"
        f"用户查询：{t_raw}"
    )
    try:
        resp = llm.invoke(prompt)
        intent = str(getattr(resp, "content", resp)).strip().lower()
        return intent if intent in {"retrieval", "tool", "direct"} else "direct"
    except Exception as e:
        logging.warning("LLM意图识别失败：%s", e)
        return "direct"


def _get_vector_store() -> Optional[FAISS]:
    """知识库向量检索句柄：惰性加载 FAISS 索引，失败返回 None"""
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE
    try:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v4", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        _VECTOR_STORE = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        logging.info("知识库索引加载成功")
        return _VECTOR_STORE
    except Exception as e:
        logging.warning("知识库索引加载失败：%s", e)
        return None


def _kb_answer(question: str) -> str:
    """RAG 问答：检索两条相关文档，严格依据 Content 字段作答"""
    try:
        vs = _get_vector_store()
        context = ""
        if vs is not None:
            docs = vs.similarity_search(question, k=2)
            context = "\n\n".join(
                f"Source: {d.metadata}\nContent: {d.page_content}" for d in docs
            )
        prompt = (
            "你是一个严谨的客服问答助手。若有参考资料，仅依据其Content字段作答。\n"
            f"参考资料：\n{context}\n\n问题：{question}\n"
        )
        resp = llm.invoke(prompt)
        answer = str(getattr(resp, "content", resp)).strip()
        return f"答：{answer}" if answer else "答：抱歉，暂未检索到相关信息。"
    except Exception as e:
        logging.error("知识库回答异常：%s", e)
        return "答：系统繁忙，请稍后再试。"


def _parse_order_id(text: str) -> Optional[str]:
    """订单号提取：支持含或不含 # 的纯数字（3-20 位）"""
    m = re.search(r"#?\d{3,20}", text)
    if not m:
        return None
    s = m.group(0)
    return s if s.startswith("#") else f"#{s}"


# Nodes
def route_query(state: State) -> Dict[str, Any]:
    """入口节点：识别意图并给出演示友好文案"""
    user_text = _clean_input(state.get("query", ""))
    step = None
    try:
        decision = router_llm.invoke(
            [
                SystemMessage(content="请依据用户请求在 course（课程检索）、order（订单查询）、general（直接答复）中选择。只返回一步。"),
                HumanMessage(content=user_text),
            ]
        )
        step = getattr(decision, "step", None)
    except Exception as e:
        logging.warning("结构化路由失败：%s", e)

    if step == "course":
        logging.info("用户输入：%s | 识别意图：retrieval", user_text)
        return {"intro": "识别为课程知识检索，将进入 RAG 流程。", "intent": "retrieval"}
    if step == "order":
        oid = _parse_order_id(user_text)
        msg = f"识别为订单查询，订单号：{oid or '未检测到'}。将进入状态查询流程。"
        logging.info("用户输入：%s | 识别意图：tool", user_text)
        return {"intro": msg, "intent": "tool"}
    if step == "general":
        try:
            msg = llm.invoke(f"请简要回答用户问题：{user_text}")
            content = str(getattr(msg, "content", msg))
        except Exception as e:
            logging.warning("LLM简答失败：%s", e)
            content = "已收到您的问题，我们将尽快处理。"
        logging.info("用户输入：%s | 识别意图：direct", user_text)
        return {"intro": content, "intent": "direct"}

    intent = _llm_detect_intent(user_text)
    logging.info("用户输入：%s | 识别意图：%s", user_text, intent)
    if intent == "retrieval":
        return {"intro": "识别为课程知识检索，将进入 RAG 流程。", "intent": intent}
    if intent == "tool":
        oid = _parse_order_id(user_text)
        msg = f"识别为订单查询，订单号：{oid or '未检测到'}。将进入状态查询流程。"
        return {"intro": msg, "intent": intent}
    try:
        msg = llm.invoke(f"请简要回答用户问题：{user_text}")
        content = str(getattr(msg, "content", msg))
    except Exception as e:
        logging.warning("LLM简答失败：%s", e)
        content = "已收到您的问题，我们将尽快处理。"
    return {"intro": content, "intent": intent}


def decide_next(state: State) -> str:
    """分支节点：依据 intent 决定后续节点"""
    return state.get("intent", "direct")


def course_answer(state: State) -> Dict[str, Any]:
    """检索节点：执行 RAG 并返回格式化答案"""
    question = state.get("query", "")
    answer = _kb_answer(question)
    return {"kb_answer": answer}


def getdb(order_text: str) -> Dict[str, Any]:
    """订单查询封装：返回 mock 与安全 SQL（参数化）"""
    # mock数据（示例）
    oid = _parse_order_id(order_text) or "#20251114001"
    mock = {
        "order_id": oid,
        "status": "processing",
        "amount": 199.0,
        "timeline": [
            "2025-11-10 申请",
            "2025-11-12 审核中",
            "2025-11-15 等待支付确认",
        ],
    }

    # 安全 SQL（参数化，避免注入；执行时转换为 SQLite 的 ? 占位符）
    sql = (
        "SELECT order_id, status, amount, updated_at "
        "FROM orders WHERE order_id = %s LIMIT 1"
    )
    params = [oid.lstrip("#")]  # 去掉前缀“#”，仅作为参数传递
    return {"mock": mock, "sql": sql, "params": params}


def _execute_order_sql(sql: str, params: list) -> Optional[Dict[str, Any]]:
    """执行订单 SQL：从 `ORDERS_DB_PATH` 指定的 SQLite 数据库读取一条记录"""
    db_path = os.getenv("ORDERS_DB_PATH")
    if not db_path:
        return None
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql.replace("%s", "?"), params)
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {
                "order_id": str(row[0]),
                "status": str(row[1]),
                "amount": float(row[2]) if row[2] is not None else None,
                "updated_at": str(row[3]) if row[3] is not None else None,
            }
        return None
    except Exception as e:
        logging.warning("订单SQL执行失败：%s", e)
        return None

def _format_order_nlg(item: Dict[str, Any]) -> str:
    """订单结果转自然语言：状态/金额/更新时间/时间线"""
    oid = item.get("order_id") or ""
    status = item.get("status") or "未知"
    amount = item.get("amount")
    updated_at = item.get("updated_at")
    parts = []
    parts.append(f"订单 {oid} 当前状态：{status}")
    if amount is not None:
        parts.append(f"金额：{amount} 元")
    if updated_at:
        parts.append(f"最近更新时间：{updated_at}")
    timeline = item.get("timeline")
    if isinstance(timeline, list) and timeline:
        parts.append("处理时间线：" + "，".join(timeline))
    return "；".join(parts)

def order_answer(state: State) -> Dict[str, Any]:
    """订单查询节点：执行 SQL（或使用 mock）并生成自然语言结果"""
    payload = getdb(state.get("query", ""))
    sql_text = payload.get("sql")
    params = payload.get("params")
    result = _execute_order_sql(sql_text, params)
    if result is None:
        result = payload.get("mock", {})
    s = _format_order_nlg(result)
    return {"order_summary": s}

def general_answer(state: State) -> Dict[str, Any]:
    """直接答复节点：对非检索/非订单问题给出简短回应"""
    user_text = state.get("query", "")
    try:
        msg = llm.invoke(f"请简要回答用户问题：{user_text}")
        content = str(getattr(msg, "content", msg))
    except Exception as e:
        logging.warning("LLM简答失败：%s", e)
        content = "已收到您的问题，我们将尽快处理。"
    return {"direct_answer": content}




def main():
    """主入口

    功能说明：读取用户输入，做基本校验与清洗，进入工作流并打印结果。
    参数：无
    返回值：无
    """
    def llm_call_router(state: State):
        """Route the input to the appropriate node"""

        decision = router_llm.invoke(
            [
                SystemMessage(
                    content="Route the input to course, order, or general based on the user's request."
                ),
                HumanMessage(content=state.get("query", "")),
            ]
        )

        return {"intent": {"course": "retrieval", "order": "tool", "general": "direct"}.get(decision.step, "direct")}

    try:
        raw = input("请输入您的问题或订单号：").strip()
    except Exception:
        raw = ""
    cleaned = _clean_input(raw)
    if not cleaned:
        print("输入为空，请重新尝试。")
        return

    workflow = StateGraph(State)
    workflow.add_node("route_query", route_query)
    workflow.add_node("course_answer", course_answer)
    workflow.add_node("order_answer", order_answer)
    workflow.add_node("general_answer", general_answer)

    workflow.add_edge(START, "route_query")
    workflow.add_conditional_edges(
        "route_query", decide_next, {"retrieval": "course_answer", "tool": "order_answer", "direct": "general_answer"}
    )
    workflow.add_edge("course_answer", END)
    workflow.add_edge("order_answer", END)
    workflow.add_edge("general_answer", END)

    # Compile & Invoke
    chain = workflow.compile()
    state = chain.invoke({"query": cleaned})

    print("初步结果：")
    print(state.get("intro", "(无)"))
    print("\n--- --- ---\n")

    intent = state.get("intent", "direct")
    if intent == "retrieval":
        print("知识库解答：")
        print(state.get("kb_answer", "(无)"))
    elif intent == "tool":
        print("订单查询结果：")
        print(state.get("order_summary", "(无)"))
    else:
        print("直接答复：")
        print(state.get("direct_answer", "(无)"))


if __name__ == "__main__":
    main()
"""
演示用问答路由器（RAG + 订单查询）

核心：
- 意图路由：retrieval | tool | direct
- 检索问答（RAG）：仅依据检索到文档的 Content 字段作答
- 订单查询：基于安全 SQL 的执行结果生成自然语言说明

运行：
- 在终端执行 `python tests/router.py` 后按提示输入

环境变量：
- `DASHSCOPE_API_KEY`：DashScope 的 API Key，用于嵌入与模型调用
- `ORDERS_DB_PATH`：SQLite 数据库文件路径，用于订单查询（可选；未配置时使用 mock）
"""
