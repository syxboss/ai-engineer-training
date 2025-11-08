# 初始化向量存储
from tabnanny import check
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.messages.utils import get_buffer_string
from typing import List
import uuid
import os
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage

# 设置DashScope API密钥
# os.environ["DASHSCOPE_API_KEY"] = "sk-your-dashscope-key"  # 请替换为你的实际API密钥

# 初始化向量存储
recall_vector_store = InMemoryVectorStore(DashScopeEmbeddings())

# 初始化模型和提示词
model = ChatTongyi(
    model="qwen-max",  # 通义千问大模型
    temperature=0.7,
    streaming=True
)

# 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是一个有记忆能力的助手。请根据用户的问题和下面提供的记忆内容回答。
    如果记忆中有相关信息，请利用这些信息来回答。
    如果需要保存新的记忆，请使用save_recall_memory工具。
    
    {recall_memory}
    """),
    MessagesPlaceholder(variable_name="messages")
])

# 定义对记忆进行存储和检索的核心函数
def get_user_id(config: RunnableConfig) -> str:
    """从RunnableConfig中提取用户ID"""
    user_id = config["configurable"].get("user_id")
    return user_id

@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """将用户记忆保存到向量存储中"""
    user_id = get_user_id(config)
    document = Document(
        page_content=memory,
        id = str(uuid.uuid4()),
        metadata={"user_id": user_id}
    )
    recall_vector_store.add_documents([document])
    return memory

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """检索用户记忆"""
    user_id = get_user_id(config)
    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id
    
    documents = recall_vector_store.similarity_search(
        query=query,
        k = 3,
        filter=_filter_function
    )
    return [doc.page_content for doc in documents]

# 声明 State 用于存储对话相关的记忆
class State(MessagesState):
    """对话状态"""
    recall_memories: List[str]

# 处理当前状态并使用 LLM 生成回复
def agent(state: State) -> State:
    """处理当前状态并生成回复"""
    # 从状态中提取记忆
    model_with_tools = model.bind_tools([save_recall_memory])
    
    # 准备记忆字符串
    recall_str = ""
    if "recall_memories" in state and state["recall_memories"]:
        recall_str = "\n".join(state["recall_memories"])
    
    # 使用提示词模板和模型生成回复
    response = prompt.invoke({
        "messages": state["messages"],
        "recall_memory": recall_str
    })
    
    # 使用绑定了工具的模型处理
    prediction = model_with_tools.invoke(response)
    
    # 返回更新后的状态
    return {
        "messages": state["messages"] + [prediction],
    }

# 添加记忆保存节点
def save_memory(state: State, config: RunnableConfig) -> State:
    """保存当前对话内容到记忆中"""
    # 获取最后一条用户消息
    messages = state["messages"]
    if len(messages) >= 2:  # 确保有用户消息和AI回复
        user_msg = None
        ai_msg = None
        
        # 找到最近的用户消息和AI回复
        for msg in reversed(messages):
            if hasattr(msg, "role"):
                if msg.role == "user" and user_msg is None:
                    user_msg = msg.content
                elif msg.role == "assistant" and ai_msg is None:
                    ai_msg = msg.content
            elif isinstance(msg, dict):
                if msg.get("role") == "user" and user_msg is None:
                    user_msg = msg.get("content", "")
                elif msg.get("role") == "assistant" and ai_msg is None:
                    ai_msg = msg.get("content", "")
                    
            if user_msg and ai_msg:
                break
                
        if user_msg and ai_msg:
            # 构建记忆内容
            memory_content = f"用户问: {user_msg}\n助手答: {ai_msg}"
            # 保存到向量存储
            save_recall_memory.invoke(memory_content, config)
            
    return {}

# 加载与当前对话相关的记忆
def load_memories(state: State, config: RunnableConfig) -> State:
    """加载与当前对话相关的记忆"""
    # 获取当前对话内容字符串
    convo_str = get_buffer_string(state["messages"])
    convo_str = convo_str[:800]
    recall_memories = search_recall_memories.invoke(convo_str, config)
    return {
        "recall_memories": recall_memories,
    }
def route_tools(state: State):
    """根据最后一条消息决定下一步操作"""
    # 获取最后一条消息
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return END

# 创建状态图并添加节点
builder = StateGraph(State)
builder.add_node("load_memories", load_memories) # 加载记忆节点
builder.add_node("agent", agent) # 处理消息节点
builder.add_node("save_memory", save_memory) # 保存记忆节点
tools = [save_recall_memory, search_recall_memories]
builder.add_node("tools", ToolNode(tools)) # 工具节点

# 添加边
# START -> load_memories -> agent -> save_memory -> route_tools
builder.add_edge(START, "load_memories") # 从START开始加载记忆
builder.add_edge("load_memories", "agent") # 加载记忆后处理消息
builder.add_edge("agent", "save_memory") # 处理消息后保存记忆
builder.add_conditional_edges(
    "save_memory",
    route_tools,  # 保存记忆后路由工具
    ["tools", END]  # 如果有工具调用，路由到tools节点；否则路由到END
) # 保存记忆后路由工具
builder.add_edge("tools", "agent") # 工具节点处理完成后返回agent节点

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)

# 运行配置：设置线程与用户ID，供记忆检索使用
config = {
    "configurable": {
        "thread_id": "thread-1",
        "user_id": "user-1",
    }
}


# 简单的流式输出处理器：打印每一步生成的消息或记忆
def get_stream_chunk(chunk):
    for node, update in chunk.items():
        if update is None:
            continue
        # 打印消息（若有）
        msgs = update.get("messages")
        if msgs:
            last = msgs[-1]
            content = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else None)
            if content:
                print(content)
        # 打印检索到的记忆（若有）
        if "recall_memories" in update and update["recall_memories"]:
            print(f"[检索到的记忆] {update['recall_memories']}")

# 第一轮对话
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "我喜欢吃苹果"}]},
    config = config
):
    get_stream_chunk(chunk)

# 第二轮询问我喜欢吃什么
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "我喜欢吃什么"}]},
    config = config
):
    get_stream_chunk(chunk)
