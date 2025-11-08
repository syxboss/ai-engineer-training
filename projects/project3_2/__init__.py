"""
混合RAG系统包
提供完整的RAG（检索增强生成）解决方案

主要组件：
- config: 配置管理
- models: 数据模型定义
- embedding: 文本嵌入服务
- retrieval: 检索功能（向量、关键词、混合）
- graph_reasoning: 图谱推理
- hybrid_rag_system: 主系统集成
"""

from .config import get_config, update_global_config, RAGConfig
from .models import (
    Entity, Relationship, Document, RetrievalResult, GraphResult,
    QAResult, ScoringResult, EmbeddingResult, KeywordExtractionResult,
    SystemMetrics, ConfidenceLevel, RetrievalSource, EntityType, RelationType
)
from .embedding import (
    QwenEmbedding, EmbeddingManager, get_embedding_manager,
    encode_text, encode_texts, calculate_similarity, calculate_similarities
)
from .retrieval import (
    KeywordExtractor, VectorRetriever, KeywordRetriever, HybridRetriever,
    create_vector_retriever, create_keyword_retriever, create_hybrid_retriever,
    extract_keywords
)
from .graph_reasoning import (
    GraphManager, EntityExtractor, GraphBuilder, GraphReasoner, GraphRAGIntegrator,
    create_graph_manager, create_graph_integrator
)
from .hybrid_rag_system import ImprovedHybridRAGSystem, create_hybrid_rag_system

# 版本信息
__version__ = "2.0.0"
__author__ = "AI Engineering Team"
__description__ = "Improved Hybrid RAG System with Modular Architecture"

# 公共接口
__all__ = [
    # 配置
    "get_config", "update_global_config", "RAGConfig",
    
    # 数据模型
    "Entity", "Relationship", "Document", "RetrievalResult", "GraphResult",
    "QAResult", "ScoringResult", "EmbeddingResult", "KeywordExtractionResult",
    "SystemMetrics", "ConfidenceLevel", "RetrievalSource", "EntityType", "RelationType",
    
    # 嵌入服务
    "QwenEmbedding", "EmbeddingManager", "get_embedding_manager",
    "encode_text", "encode_texts", "calculate_similarity", "calculate_similarities",
    
    # 检索功能
    "KeywordExtractor", "VectorRetriever", "KeywordRetriever", "HybridRetriever",
    "create_vector_retriever", "create_keyword_retriever", "create_hybrid_retriever",
    "extract_keywords",
    
    # 图谱推理
    "GraphManager", "EntityExtractor", "GraphBuilder", "GraphReasoner", "GraphRAGIntegrator",
    "create_graph_manager", "create_graph_integrator",
    
    # 主系统
    "ImprovedHybridRAGSystem", "create_hybrid_rag_system"
]

# 快速开始函数
def quick_start(neo4j_driver=None, llm_json=None, llm_text=None, use_qwen_embedding=True):
    """快速启动RAG系统
    
    Args:
        neo4j_driver: Neo4j数据库驱动
        llm_json: 结构化输出LLM
        llm_text: 文本生成LLM
        use_qwen_embedding: 是否使用通义千问嵌入
        
    Returns:
        ImprovedHybridRAGSystem实例
    """
    if not llm_json or not llm_text:
        raise ValueError("必须提供LLM实例")
    
    return create_hybrid_rag_system(neo4j_driver, llm_json, llm_text, use_qwen_embedding)

# 系统信息
def get_system_info():
    """获取系统信息"""
    return {
        "name": "Improved Hybrid RAG System",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "components": [
            "Configuration Management",
            "Data Models",
            "Text Embedding (Qwen + SentenceTransformer)",
            "Multi-Modal Retrieval (Vector + Keyword + Graph)",
            "Knowledge Graph Reasoning",
            "Error Propagation Guard",
            "Performance Monitoring"
        ],
        "features": [
            "Modular Architecture",
            "Chinese Text Optimization",
            "Hybrid Retrieval Strategy",
            "Graph-Enhanced Reasoning",
            "Configurable Parameters",
            "Comprehensive Logging",
            "Error Handling & Recovery"
        ]
    }

# 配置验证
def validate_environment():
    """验证环境配置"""
    config = get_config()
    
    issues = []
    
    # 检查API密钥
    if config.dashscope_api_key == "your-api-key-here":
        issues.append("未设置DASHSCOPE_API_KEY环境变量")
    
    # 检查配置有效性
    if not config.validate_config():
        issues.append("配置参数验证失败")
    
    if issues:
        return {
            "status": "warning",
            "issues": issues,
            "message": "发现配置问题，系统可能无法正常工作"
        }
    else:
        return {
            "status": "ok",
            "message": "环境配置正常"
        }

# 模块导入时的初始化检查
import logging
logger = logging.getLogger(__name__)

try:
    # 验证环境
    env_status = validate_environment()
    if env_status["status"] == "warning":
        logger.warning(f"环境配置警告: {env_status['message']}")
        for issue in env_status["issues"]:
            logger.warning(f"  - {issue}")
    else:
        logger.info("✅ 混合RAG系统包加载完成，环境配置正常")
        
except Exception as e:
    logger.error(f"包初始化时发生错误: {e}")