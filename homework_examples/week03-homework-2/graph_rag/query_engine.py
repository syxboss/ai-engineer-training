import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.prompts import PromptTemplate
from . import config


# --- 全局变量 ---
_rag_query_engine = None
_kg_query_engine = None


# --- RAG 索引初始化 ---
def _initialize_rag_index():
    """初始化或加载 RAG 的向量索引"""
    global _rag_query_engine
    if not os.path.exists(config.INDEX_DIR):
        print("未找到向量索引，正在从文件创建...")
        documents = SimpleDirectoryReader(
            input_files=[config.COMPANY_DOC_PATH]
        ).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=config.INDEX_DIR)
        print(f"向量索引已创建并保存在 '{config.INDEX_DIR}'。")
    else:
        print(f"从 '{config.INDEX_DIR}' 加载现有向量索引...")
        storage_context = StorageContext.from_defaults(persist_dir=config.INDEX_DIR)
        index = load_index_from_storage(storage_context)
        print("向量索引加载成功。")
    
    _rag_query_engine = index.as_query_engine(similarity_top_k=2)


# --- 知识图谱查询引擎初始化 ---
def _initialize_kg_query_engine():
    """初始化知识图谱查询引擎"""
    global _kg_query_engine
    print("正在连接到 Neo4j 并初始化知识图谱查询引擎...")
    graph_store = Neo4jGraphStore(
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        url=config.NEO4J_URI,
        database=config.NEO4J_DATABASE,
    )
    
    # 这个查询引擎能将自然语言转换为 Cypher 查询
    _kg_query_engine = KnowledgeGraphQueryEngine(
        storage_context=StorageContext.from_defaults(graph_store=graph_store),
        llm=config.Settings.llm,
        verbose=True, # 打印生成的 Cypher 查询，便于调试
    )


# --- 多跳查询主函数 ---
def multi_hop_query(question: str):
    """
    执行多跳查询：RAG -> KG -> LLM
    """
    if not _rag_query_engine or not _kg_query_engine:
        raise RuntimeError("查询引擎未初始化。请先运行初始化函数。")

    reasoning_path = []

    # 1. RAG 检索：识别问题中的核心实体
    # 我们使用一个简单的提示来让 LLM 提取公司名称
    entity_extraction_prompt = PromptTemplate(
        "从以下问题中提取出公司或机构的名称：'{question}'\n"
        "只返回名称，不要添加任何其他文字。"
    )
    formatted_prompt = entity_extraction_prompt.format(question=question)
    entity_name_response = config.Settings.llm.complete(formatted_prompt)
    entity_name = entity_name_response.text.strip()
    
    reasoning_path.append(f"步骤 1: 从问题 '{question}' 中识别出核心实体 -> '{entity_name}'")

    # 2. 图谱查询：使用识别出的实体查询图谱
    # 针对“最大股东”这类问题，我们直接构造 Cypher 查询以获得更精确的结果
    # 这是一个“Cypher + LLM”协同的例子
    cypher_query = ""
    if "最大股东" in question or "控股" in question:
        cypher_query = f"""
        MATCH (shareholder:Entity)-[r:HOLDS_SHARES_IN]->(company:Entity {{name: '{entity_name}'}})
        RETURN shareholder.name AS shareholder, r.share_percentage AS percentage
        ORDER BY percentage DESC
        LIMIT 1
        """
        reasoning_path.append(f"步骤 2: 识别到关键词'最大股东'，构造精确 Cypher 查询在图谱中查找。")
        reasoning_path.append(f"   - Cypher 查询: {cypher_query.strip()}")
        
        # 直接执行 Cypher
        graph_store = _kg_query_engine.storage_context.graph_store
        graph_response = graph_store.query(cypher_query)
        kg_result_text = str(graph_response)

    else:
        # 对于其他通用问题，使用 LlamaIndex 的自然语言转 Cypher 功能
        reasoning_path.append(f"步骤 2: 未识别到特定模式，使用 LLM 将自然语言转换为 Cypher 查询。")
        kg_response = _kg_query_engine.query(f"查询与 '{entity_name}' 相关的信息")
        kg_result_text = kg_response.response
        # kg_response.metadata['cypher_query'] 中包含了生成的查询
        if 'cypher_query' in kg_response.metadata:
            reasoning_path.append(f"   - 生成的 Cypher 查询: {kg_response.metadata['cypher_query'].strip()}")

    reasoning_path.append(f"   - 图谱查询结果: {kg_result_text}")

    # 3. RAG 补充信息
    rag_response = _rag_query_engine.query(f"提供关于 '{entity_name}' 的详细信息。")
    rag_context = "\n\n".join([node.get_content() for node in rag_response.source_nodes])
    reasoning_path.append(f"步骤 3: 通过 RAG 检索关于 '{entity_name}' 的背景文档信息。")
    reasoning_path.append(f"   - RAG 检索到的上下文: {rag_context[:200]}...") # 仅显示部分

    # 4. LLM 生成最终回答
    final_answer_prompt = PromptTemplate(
        "你是一个专业的金融分析师。请根据以下信息，以清晰、简洁的语言回答用户的问题。\n"
        "--- 用户问题 ---\n{question}\n\n"
        "--- 知识图谱查询结果 ---\n{kg_result}\n\n"
        "--- 相关文档信息 ---\n{rag_context}\n\n"
        "--- 最终回答 ---\n"
    )
    
    formatted_prompt = final_answer_prompt.format(
        question=question,
        kg_result=kg_result_text,
        rag_context=rag_context
    )
    
    reasoning_path.append("步骤 4: 综合图谱结果和文档信息，由 LLM 生成最终的自然语言回答。")
    final_response = config.Settings.llm.complete(formatted_prompt)
    final_answer = final_response.text

    return {
        "final_answer": final_answer,
        "reasoning_path": reasoning_path
    }


# --- 初始化所有引擎 ---
def initialize_all():
    _initialize_rag_index()
    _initialize_kg_query_engine()


# 在模块加载时执行初始化
initialize_all()