"""
检索过程模块
包含混合RAG系统的检索相关功能
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import *
from models import Document, RetrievalResult, GraphResult
from embedding import EmbeddingManager
from graph_reasoning import Neo4jManager


class HybridRetrievalSystem:
    """混合检索系统"""
    
    def __init__(self, embedding_manager: EmbeddingManager, neo4j_manager: Neo4jManager):
        """
        初始化混合检索系统
        
        Args:
            embedding_manager: 嵌入管理器
            neo4j_manager: Neo4j管理器
        """
        self.embedding_manager = embedding_manager
        self.neo4j_manager = neo4j_manager
        self.logger = logging.getLogger(__name__)
    
    def vector_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[RetrievalResult]:
        """
        向量检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        try:
            results = self.embedding_manager.vector_search(query, top_k)
            # EmbeddingManager已经返回RetrievalResult对象，直接返回
            return results
        except Exception as e:
            self.logger.error(f"向量检索失败: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[RetrievalResult]:
        """
        关键词检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        try:
            results = self.embedding_manager.keyword_search(query, top_k)
            # EmbeddingManager已经返回RetrievalResult对象，直接返回
            return results
        except Exception as e:
            self.logger.error(f"关键词检索失败: {e}")
            return []
    
    def graph_search(self, query: str, max_depth: int = GRAPH_MAX_DEPTH) -> List[GraphResult]:
        """
        图检索
        
        Args:
            query: 查询文本
            max_depth: 最大搜索深度
            
        Returns:
            图检索结果列表
        """
        try:
            results = self.neo4j_manager.graph_search(query, max_depth)
            return [GraphResult(
                entity=result['entity'],
                relationships=result['relationships'],
                score=result['score']
            ) for result in results]
        except Exception as e:
            self.logger.error(f"图检索失败: {e}")
            return []
    
    def calculate_joint_score(self, vector_results: List[RetrievalResult], 
                            keyword_results: List[RetrievalResult],
                            graph_result: GraphResult) -> Dict[str, Any]:
        """
        改进的联合评分机制
        """
        # 收集所有文档
        all_docs = {}
        
        # 向量检索结果
        for result in vector_results:
            doc_id = result.document.id
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    'document': result.document,
                    'vector_score': 0.0,
                    'keyword_score': 0.0,
                    'graph_score': 0.0
                }
            all_docs[doc_id]['vector_score'] = result.score
        
        # 关键词检索结果
        for result in keyword_results:
            doc_id = result.document.id
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    'document': result.document,
                    'vector_score': 0.0,
                    'keyword_score': 0.0,
                    'graph_score': 0.0
                }
            all_docs[doc_id]['keyword_score'] = result.score
        
        # 图谱推理分数（基于文档内容与图谱关系的匹配度）
        graph_confidence = graph_result.confidence
        for doc_id in all_docs:
            # 简单的图谱相关性评分
            all_docs[doc_id]['graph_score'] = graph_confidence
        
        # 计算综合分数
        final_results = []
        for doc_id, scores in all_docs.items():
            joint_score = (
                scores['vector_score'] * RETRIEVAL_VECTOR_WEIGHT +
                scores['keyword_score'] * RETRIEVAL_KEYWORD_WEIGHT +
                scores['graph_score'] * RETRIEVAL_GRAPH_WEIGHT
            )
            
            final_results.append({
                'document': scores['document'],
                'joint_score': joint_score,
                'vector_score': scores['vector_score'],
                'keyword_score': scores['keyword_score'],
                'graph_score': scores['graph_score']
            })
        
        # 排序
        final_results.sort(key=lambda x: x['joint_score'], reverse=True)
        
        # 计算整体置信度
        if final_results:
            max_score = final_results[0]['joint_score']
            overall_confidence = min(max_score, 1.0)
        else:
            overall_confidence = 0.0
        
        return {
            'results': final_results,
            'overall_confidence': overall_confidence,
            'vector_count': len(vector_results),
            'keyword_count': len(keyword_results),
            'graph_confidence': graph_result.confidence
        }
    
    def error_propagation_guard(self, results: Dict[str, Any], 
                              vector_results: List[RetrievalResult],
                              graph_result: GraphResult) -> Dict[str, Any]:
        """
        改进的错误传播防护
        """
        warnings = []
        
        # 检查整体置信度
        if results['overall_confidence'] < 0.3:  # error_propagation_threshold
            warnings.append("整体置信度过低，可能存在错误传播风险")
        
        # 检查各模块置信度
        if not vector_results:
            warnings.append("向量检索无结果，建议检查embedding质量")
        
        if results['keyword_count'] == 0:
            warnings.append("关键词检索无结果，建议优化分词策略")
        
        if graph_result.confidence < 0.3:
            warnings.append("图谱推理置信度过低，建议人工验证")
        
        # 置信度等级
        confidence = results['overall_confidence']
        if confidence >= 0.7:
            confidence_level = "high"
        elif confidence >= 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return {
            **results,
            'confidence_level': confidence_level,
            'warnings': warnings
        }

    def hybrid_search(self, query: str, 
                     vector_weight: float = RETRIEVAL_VECTOR_WEIGHT,
                     keyword_weight: float = RETRIEVAL_KEYWORD_WEIGHT,
                     graph_weight: float = RETRIEVAL_GRAPH_WEIGHT,
                     top_k: int = RETRIEVAL_TOP_K) -> Dict[str, Any]:
        """
        混合检索
        
        Args:
            query: 查询文本
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            graph_weight: 图检索权重
            top_k: 返回结果数量
            
        Returns:
            混合检索结果
        """
        try:
            # 执行各种检索
            vector_results = self.vector_search(query, top_k)
            keyword_results = self.keyword_search(query, top_k)
            graph_results = self.graph_search(query)
            
            # 创建GraphResult对象用于联合评分
            if graph_results:
                graph_result = GraphResult(
                    entity=graph_results[0].entity if graph_results else "",
                    relationships=sum([r.relationships for r in graph_results], []),
                    score=sum([r.score for r in graph_results]) / len(graph_results) if graph_results else 0.0
                )
            else:
                graph_result = GraphResult(entities=[], relationships=[], confidence=0.0, reasoning_path=[])
            
            # 使用联合评分机制
            scoring_results = self.calculate_joint_score(vector_results, keyword_results, graph_result)
            
            # 错误传播防护
            final_results = self.error_propagation_guard(scoring_results, vector_results, graph_result)
            
            # 转换为标准格式
            results = []
            for result in final_results['results'][:top_k]:
                results.append({
                    'content': result['document'].content,
                    'score': result['joint_score'],
                    'source': 'hybrid'
                })
            
            return {
                'results': results,
                'vector_count': len(vector_results),
                'keyword_count': len(keyword_results),
                'graph_count': len(graph_results),
                'total_count': len(results),
                'confidence': final_results['overall_confidence'],
                'confidence_level': final_results['confidence_level'],
                'warnings': final_results['warnings']
            }
            
        except Exception as e:
            self.logger.error(f"混合检索失败: {e}")
            return {
                'results': [],
                'vector_count': 0,
                'keyword_count': 0,
                'graph_count': 0,
                'total_count': 0,
                'confidence': 0.0,
                'confidence_level': 'low',
                'warnings': ['检索过程中出现错误']
            }
    
    def multi_hop_qa(self, question: str, max_hops: int = GRAPH_MAX_HOPS) -> Dict[str, Any]:
        """
        多跳问答
        
        Args:
            question: 问题
            max_hops: 最大跳数
            
        Returns:
            多跳问答结果
        """
        try:
            # 首先进行混合检索获取初始上下文
            initial_results = self.hybrid_search(question)
            
            # 从图中进行多跳推理
            graph_reasoning_results = self.neo4j_manager.multi_hop_reasoning(question, max_hops)
            
            # 合并结果
            all_context = []
            
            # 添加检索结果
            for result in initial_results['results']:
                all_context.append(result['content'])
            
            # 添加图推理结果
            for result in graph_reasoning_results:
                context = f"推理路径: {' -> '.join(result['path'])}, 置信度: {result['confidence']:.3f}"
                all_context.append(context)
            
            return {
                'question': question,
                'context': all_context,
                'retrieval_results': initial_results,
                'reasoning_results': graph_reasoning_results,
                'total_context_items': len(all_context)
            }
            
        except Exception as e:
            self.logger.error(f"多跳问答失败: {e}")
            return {
                'question': question,
                'context': [],
                'retrieval_results': {'results': [], 'vector_count': 0, 'keyword_count': 0, 'graph_count': 0, 'total_count': 0},
                'reasoning_results': [],
                'total_context_items': 0
            }