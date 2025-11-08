import os

# 导入工具和模式定义
from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

# 导入状态定义
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
# 导入配置和提示词
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
# 导入工具函数
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

# 加载环境变量
load_dotenv()

# 检查API密钥是否设置
if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# 用于Google搜索API的客户端
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# LangGraph节点定义
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """
    LangGraph节点：基于用户问题生成搜索查询
    
    使用Gemini 2.0 Flash根据用户问题创建优化的网络研究搜索查询
    
    Args:
        state: 包含用户问题的当前图状态
        config: 运行配置，包括LLM提供商设置
        
    Returns:
        包含状态更新的字典，包括包含生成查询的search_query键
    """
    configurable = Configuration.from_runnable_config(config)

    # 检查自定义初始搜索查询计数
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # 初始化Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # 格式化提示词
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # 生成搜索查询
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """
    LangGraph节点：将搜索查询发送到网络研究节点
    
    用于生成n个网络研究节点，每个搜索查询对应一个节点
    这实现了并行搜索处理
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """
    LangGraph节点：使用原生Google搜索API工具执行网络研究
    
    结合Gemini 2.0 Flash使用原生Google搜索API工具执行网络搜索
    
    Args:
        state: 包含搜索查询和研究循环计数的当前图状态
        config: 运行配置，包括搜索API设置
        
    Returns:
        包含状态更新的字典，包括sources_gathered、research_loop_count和web_research_results
    """
    # 配置
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # 使用google genai客户端，因为langchain客户端不返回基础元数据
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # 将URL解析为短URL以节省令牌和时间
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    # 获取引用并将其添加到生成的文本中
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """
    LangGraph节点：识别知识缺口并生成潜在的后续查询
    
    分析当前摘要以识别需要进一步研究的领域，并生成潜在的后续查询。
    使用结构化输出以JSON格式提取后续查询。
    
    Args:
        state: 包含运行摘要和研究主题的当前图状态
        config: 运行配置，包括LLM提供商设置
        
    Returns:
        包含状态更新的字典，包括包含生成的后续查询的search_query键
    """
    configurable = Configuration.from_runnable_config(config)
    # 增加研究循环计数并获取推理模型
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # 格式化提示词
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # 初始化推理模型
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """
    LangGraph路由函数：确定研究流程中的下一步
    
    通过决定是继续收集信息还是基于配置的最大研究循环数完成摘要来控制研究循环
    
    Args:
        state: 包含研究循环计数的当前图状态
        config: 运行配置，包括max_research_loops设置
        
    Returns:
        指示要访问的下一个节点的字符串字面量（"web_research"或"finalize_answer"）
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """
    LangGraph节点：完成研究摘要
    
    通过去重和格式化来源来准备最终输出，然后将它们与运行摘要结合，
    创建一个具有适当引用的结构良好的研究报告
    
    Args:
        state: 包含运行摘要和收集的来源的当前图状态
        config: 运行配置
        
    Returns:
        包含状态更新的字典，包括包含带有来源的格式化最终摘要的running_summary键
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # 格式化提示词
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # 初始化推理模型，默认为Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # 将短URL替换为原始URL，并将所有使用的URL添加到sources_gathered中
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# 创建我们的代理图
builder = StateGraph(OverallState, config_schema=Configuration)

# 定义我们将在其间循环的节点
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# 将入口点设置为`generate_query`
# 这意味着这个节点是第一个被调用的
builder.add_edge(START, "generate_query")
# 添加条件边以在并行分支中继续搜索查询
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# 对网络研究进行反思
builder.add_edge("web_research", "reflection")
# 评估研究
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# 完成答案
builder.add_edge("finalize_answer", END)

# 编译图并命名为"pro-search-agent"
graph = builder.compile(name="pro-search-agent")
