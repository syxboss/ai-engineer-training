import asyncio
import json
import logging
import numpy as np
import os
import dashscope
import requests
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import jieba.analyse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置通义千问API密钥
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")

@dataclass
class Entity:
    name: str
    type: str
    confidence: float = 1.0

@dataclass  
class Relationship:
    source: str
    target: str
    type: str
    confidence: float = 1.0

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

@dataclass
class RetrievalResult:
    document: Document
    score: float
    source: str  # 'vector' or 'keyword' or 'graph'

@dataclass
class GraphResult:
    entities: List[str]
    relationships: List[Dict]
    confidence: float
    reasoning_path: List[str]

class QwenEmbedding:
    """通义千问文本嵌入服务"""
    
    def __init__(self, model_name="text-embedding-v3"):
        self.model_name = model_name
        
    def encode(self, texts):
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            from dashscope import TextEmbedding
            
            response = TextEmbedding.call(
                model=self.model_name,
                input=texts
            )
            
            if response.status_code == 200:
                embeddings = []
                for output in response.output['embeddings']:
                    embeddings.append(np.array(output['embedding']))
                
                return embeddings[0] if len(embeddings) == 1 else embeddings
            else:
                logger.error(f"通义千问embedding调用失败: {response}")
                # 降级到本地模型
                fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
                return fallback_model.encode(texts)
                
        except Exception as e:
            logger.error(f"通义千问embedding异常: {e}")
            # 降级到本地模型
            fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
            return fallback_model.encode(texts)

class ImprovedHybridRAGSystem:
    """
    改进的混合RAG系统
    
    主要改进：
    1. 使用通义千问text-embedding-v3提升中文理解
    2. 优化关键词提取，使用jieba分词
    3. 改进相似度计算和阈值设置
    4. 增强图谱推理逻辑
    5. 添加向量数据库内置函数对比测试
    """
    
    def __init__(self, neo4j_driver, llm_json, llm_text, use_qwen_embedding=True):
        self.driver = neo4j_driver
        self.llm_json = llm_json  # 结构化输出
        self.llm_text = llm_text  # 文本生成
        
        # 初始化向量模型
        if use_qwen_embedding:
            self.embedding_model = QwenEmbedding("text-embedding-v3")
            logger.info("✅ 使用通义千问text-embedding-v3模型")
        else:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ 使用SentenceTransformer模型")
        
        # 文档存储
        self.documents: List[Document] = []
        
        # 关键词索引
        self.keyword_index = {
            'terms': {},  # term -> [doc_indices]
            'doc_terms': {}  # doc_id -> [terms]
        }
        
        # 配置参数 - 降低阈值提升召回率
        self.config = {
            'confidence_threshold': 0.3,  # 降低置信度阈值
            'max_retrieval_results': 10,   # 增加检索结果数
            'vector_weight': 0.5,         # 提高向量检索权重
            'keyword_weight': 0.3,        # 关键词检索权重  
            'graph_weight': 0.2,          # 图谱推理权重
            'error_propagation_threshold': 0.3,  # 降低错误传播阈值
            'keyword_threshold': 0.1      # 关键词匹配阈值
        }
    
    def clear_vector_database(self):
        """清理向量数据库，删除所有文档和嵌入"""
        self.documents = []
        self.keyword_index = {
            'terms': {},  # term -> [doc_indices]
            'doc_terms': {}  # doc_id -> [terms]
        }
        logger.info("✅ 向量数据库已清理")
    
    def clear_graph_database(self):
        """清理图数据库，删除所有节点和关系"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("✅ 图数据库已清理")
    
    def clear_all_databases(self):
        """清理所有数据库"""
        self.clear_vector_database()
        self.clear_graph_database()
        logger.info("🧹 所有数据库已清理完成")
    
    def load_data_from_file(self, file_path: str) -> str:
        """从文件加载数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"文件读取失败: {e}")
            return ""
    
    async def process_text_to_documents(self, raw_text: str) -> List[Dict[str, Any]]:
        """将原始文本处理为文档"""
        # 按段落分割
        paragraphs = [p.strip() for p in raw_text.split('\n') if p.strip()]
        
        documents = []
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 10:  # 过滤太短的段落
                documents.append({
                    'id': f'doc_{i}',
                    'content': paragraph,
                    'metadata': {'source': 'file', 'paragraph_id': i}
                })
        
        return documents
    
    async def extract_relationships_from_text(self, raw_text: str) -> List[Tuple[str, str]]:
        """从文本中提取关系"""
        prompt = f"""
        从以下文本中提取公司控股关系，返回JSON格式：
        
        文本：{raw_text}
        
        返回格式：
        {{
            "relationships": [
                {{"source": "公司A", "target": "公司B", "type": "控股"}}
            ]
        }}
        
        注意：只提取明确的控股关系，包括"控股"、"持股"、"投资"等关系。
        """
        
        try:
            response = await self.llm_json.ainvoke(prompt)
            result = json.loads(response.content)
            
            relationships = []
            for rel in result.get("relationships", []):
                relationships.append((rel["source"], rel["target"]))
            
            return relationships
        except Exception as e:
            logger.error(f"关系提取失败: {e}")
            return []
    
    def extract_keywords(self, text: str) -> List[str]:
        """使用jieba提取关键词"""
        # 使用TF-IDF提取关键词
        keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False)
        
        # 添加基础分词
        words = jieba.cut(text)
        basic_words = [w for w in words if len(w) > 1 and w.isalnum()]
        
        # 合并并去重
        all_keywords = list(set(keywords + basic_words))
        return all_keywords
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """添加文档到向量数据库"""
        for doc_data in documents:
            # 创建文档对象
            doc = Document(
                id=doc_data['id'],
                content=doc_data['content'],
                metadata=doc_data['metadata']
            )
            
            # 生成向量
            doc.embedding = self.embedding_model.encode(doc.content)
            
            # 提取关键词并建立索引
            keywords = self.extract_keywords(doc.content)
            self.keyword_index['doc_terms'][doc.id] = keywords
            
            for keyword in keywords:
                if keyword not in self.keyword_index['terms']:
                    self.keyword_index['terms'][keyword] = []
                self.keyword_index['terms'][keyword].append(len(self.documents))
            
            self.documents.append(doc)
        
        logger.info(f"添加了 {len(documents)} 个文档到检索库")
    
    def vector_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        改进的向量检索：使用更好的相似度计算
        """
        if not self.documents:
            return []
        
        # 查询向量化
        query_embedding = self.embedding_model.encode(query)
        
        # 计算相似度
        doc_embeddings = np.array([doc.embedding for doc in self.documents])
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # 使用更低的阈值，提升召回率
        results = []
        for i, score in enumerate(similarities):
            if score > self.config['confidence_threshold']:
                results.append(RetrievalResult(
                    document=self.documents[i],
                    score=float(score),
                    source='vector'
                ))
        
        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"向量检索: {len(results)} 个结果")
        return results[:top_k]
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        改进的关键词检索：使用jieba分词和更灵活的匹配
        """
        if not self.documents:
            return []
        
        # 提取查询关键词
        query_keywords = self.extract_keywords(query)
        
        results = []
        for i, doc in enumerate(self.documents):
            doc_keywords = self.keyword_index['doc_terms'].get(doc.id, [])
            
            # 计算关键词匹配度
            if query_keywords and doc_keywords:
                intersection = set(query_keywords).intersection(set(doc_keywords))
                union = set(query_keywords).union(set(doc_keywords))
                
                if intersection:
                    # Jaccard相似度
                    jaccard_score = len(intersection) / len(union)
                    
                    # 考虑关键词在文档中的重要性
                    importance_score = len(intersection) / len(query_keywords)
                    
                    # 综合分数
                    final_score = (jaccard_score + importance_score) / 2
                    
                    if final_score > self.config['keyword_threshold']:
                        results.append(RetrievalResult(
                            document=doc,
                            score=final_score,
                            source='keyword'
                        ))
        
        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"关键词检索: {len(results)} 个结果")
        return results[:top_k]
    
    async def extract_entities_from_query(self, query: str) -> List[Entity]:
        """
        改进的实体提取
        """
        prompt = f"""
        从问题中提取所有相关的实体（公司名、人名等）：
        
        问题：{query}
        
        返回JSON格式：
        {{
            "entities": [
                {{"name": "实体名", "type": "Company|Person", "confidence": 0.9}}
            ]
        }}
        
        注意：尽可能提取所有可能相关的实体，包括简称和全称。
        """
        
        try:
            response = await self.llm_json.ainvoke(prompt)
            result = json.loads(response.content)
            
            entities = []
            for e in result.get("entities", []):
                entities.append(Entity(
                    name=e["name"],
                    type=e["type"],
                    confidence=e.get("confidence", 0.8)
                ))
            
            logger.info(f"提取到 {len(entities)} 个实体")
            return entities
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return []
    
    def graph_reasoning(self, entities: List[Entity], query: str) -> GraphResult:
        """
        改进的图谱推理：支持多跳查询和更灵活的匹配
        """
        if not entities:
            return GraphResult([], [], 0.0, [])
        
        entity_names = [e.name for e in entities]
        
        # 构建更灵活的查询，支持模糊匹配
        cypher_queries = []
        
        # 1. 直接关系查询
        for entity in entity_names:
            cypher_queries.append(f"""
                MATCH (a)-[r]->(b)
                WHERE a.name CONTAINS '{entity}' OR b.name CONTAINS '{entity}'
                RETURN a.name as source, type(r) as relation, b.name as target, 1.0 as confidence
            """)
        
        # 2. 多跳关系查询（2跳）
        for entity in entity_names:
            cypher_queries.append(f"""
                MATCH (a)-[r1]->(b)-[r2]->(c)
                WHERE a.name CONTAINS '{entity}' OR c.name CONTAINS '{entity}'
                RETURN a.name as source, type(r1) + '->' + type(r2) as relation, c.name as target, 0.8 as confidence
            """)
        
        all_relationships = []
        reasoning_paths = []
        
        with self.driver.session() as session:
            for cypher in cypher_queries:
                try:
                    result = session.run(cypher)
                    for record in result:
                        relationship = {
                            'source': record['source'],
                            'relation': record['relation'],
                            'target': record['target'],
                            'confidence': record['confidence']
                        }
                        all_relationships.append(relationship)
                        reasoning_paths.append(f"{record['source']} -> {record['relation']} -> {record['target']}")
                except Exception as e:
                    logger.error(f"图谱查询失败: {e}")
        
        # 去重
        unique_relationships = []
        seen = set()
        for rel in all_relationships:
            key = (rel['source'], rel['relation'], rel['target'])
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        # 计算整体置信度
        if unique_relationships:
            avg_confidence = sum(r['confidence'] for r in unique_relationships) / len(unique_relationships)
        else:
            avg_confidence = 0.0
        
        logger.info(f"图谱推理: {len(unique_relationships)} 个关系")
        
        return GraphResult(
            entities=entity_names,
            relationships=unique_relationships,
            confidence=avg_confidence,
            reasoning_path=reasoning_paths
        )
    
    def test_vector_similarity(self, query: str, documents: List[str]) -> Dict[str, Any]:
        """
        测试向量相似度计算，用于调试和优化
        """
        query_embedding = self.embedding_model.encode(query)
        doc_embeddings = [self.embedding_model.encode(doc) for doc in documents]
        
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append({
                'document_index': i,
                'document_preview': documents[i][:100] + "..." if len(documents[i]) > 100 else documents[i],
                'similarity_score': float(similarity)
            })
        
        # 排序
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {
            'query': query,
            'similarities': similarities,
            'threshold': self.config['confidence_threshold']
        }
    
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
                scores['vector_score'] * self.config['vector_weight'] +
                scores['keyword_score'] * self.config['keyword_weight'] +
                scores['graph_score'] * self.config['graph_weight']
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
        if results['overall_confidence'] < self.config['error_propagation_threshold']:
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
    
    async def multi_hop_qa(self, question: str) -> Dict[str, Any]:
        """
        改进的多跳问答
        """
        logger.info(f"开始处理问题: {question}")
        
        # 1. 实体提取
        entities = await self.extract_entities_from_query(question)
        
        # 2. 多源检索
        vector_results = self.vector_search(question, top_k=self.config['max_retrieval_results'])
        keyword_results = self.keyword_search(question, top_k=self.config['max_retrieval_results'])
        graph_result = self.graph_reasoning(entities, question)
        
        # 3. 联合评分
        scoring_results = self.calculate_joint_score(vector_results, keyword_results, graph_result)
        
        # 4. 错误传播防护
        final_results = self.error_propagation_guard(scoring_results, vector_results, graph_result)
        
        # 5. 生成最终答案
        final_answer = await self.generate_final_answer(
            question, vector_results, keyword_results, graph_result, final_results
        )
        
        return {
            'question': question,
            'answer': final_answer,
            'confidence': final_results['overall_confidence'],
            'confidence_level': final_results['confidence_level'],
            'warnings': final_results['warnings'],
            'vector_count': len(vector_results),
            'keyword_count': len(keyword_results),
            'graph_relationships': len(graph_result.relationships)
        }
    
    async def generate_final_answer(self, question: str, 
                                  vector_results: List[RetrievalResult],
                                  keyword_results: List[RetrievalResult], 
                                  graph_result: GraphResult,
                                  scoring_results: Dict[str, Any]) -> str:
        """
        改进的答案生成
        """
        # 收集上下文
        contexts = []
        
        # 向量检索上下文
        for result in vector_results[:3]:  # 取前3个
            contexts.append(f"文档内容: {result.document.content}")
        
        # 图谱推理上下文
        if graph_result.relationships:
            graph_context = "图谱关系:\n"
            for rel in graph_result.relationships[:5]:  # 取前5个关系
                graph_context += f"- {rel['source']} {rel['relation']} {rel['target']}\n"
            contexts.append(graph_context)
        
        context_text = "\n\n".join(contexts)
        
        prompt = f"""
        基于以下信息回答问题：
        
        问题：{question}
        
        上下文信息：
        {context_text}
        
        请根据提供的信息给出准确、详细的答案。如果信息不足，请说明不确定性。
        """
        
        try:
            response = await self.llm_text.ainvoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return "抱歉，无法生成答案。"

# 测试和演示函数
async def demo_improved():
    """改进版本的演示"""
    # Neo4j连接
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    
    # LLM配置
    llm_json = OpenAILLM(
        model_name="gpt-4o-mini",
        model_params={"response_format": {"type": "json_object"}}
    )
    llm_text = OpenAILLM(model_name="gpt-4o-mini")
    
    # 创建改进的系统
    system = ImprovedHybridRAGSystem(driver, llm_json, llm_text, use_qwen_embedding=True)
    
    # 清理数据库
    system.clear_all_databases()
    
    # 加载数据
    raw_text = system.load_data_from_file("company.txt")
    
    # 处理文档
    documents = await system.process_text_to_documents(raw_text)
    system.add_documents(documents)
    print(f"✅ 成功添加 {len(documents)} 个文档到向量数据库")
    
    # 构建图谱
    print("🔗 提取控股关系并构建知识图谱...")
    relationships = await system.extract_relationships_from_text(raw_text)
    await build_sample_graph_from_relationships(driver, relationships)
    print(f"✅ 成功构建包含 {len(relationships)} 个关系的知识图谱")
    
    # 测试问题
    questions = [
        "A集团的最大股东是谁？",
        "B资本控制哪些公司？", 
        "A集团有多少层级的控股关系？"
    ]
    
    for question in questions:
        print(f"\n📋 问题: {question}")
        print("-" * 40)
        
        result = await system.multi_hop_qa(question)
        
        print(f"🎯 最终答案: {result['answer']}")
        print(f"📊 整体置信度: {result['confidence']:.2f} ({result['confidence_level']})")
        
        if result['warnings']:
            print(f"⚠️  警告: {'; '.join(result['warnings'])}")
        
        print(f"🔍 检索到 {result['vector_count']} 个向量结果")
        print(f"🔗 图谱推理找到 {result['graph_relationships']} 个关系")
    
    # 向量相似度测试
    print("\n🧪 向量相似度测试:")
    test_docs = [doc['content'] for doc in documents[:5]]
    similarity_test = system.test_vector_similarity("B资本控制哪些公司", test_docs)
    
    print(f"查询: {similarity_test['query']}")
    print(f"阈值: {similarity_test['threshold']}")
    for sim in similarity_test['similarities'][:3]:
        print(f"  相似度 {sim['similarity_score']:.3f}: {sim['document_preview']}")
    
    driver.close()

async def build_sample_graph_from_relationships(driver, relationships: List[Tuple[str, str]]):
    """从关系列表构建图谱"""
    with driver.session() as session:
        for source, target in relationships:
            session.run("""
                MERGE (a:Company {name: $source})
                MERGE (b:Company {name: $target})
                MERGE (a)-[:CONTROLS]->(b)
            """, source=source, target=target)

if __name__ == "__main__":
    asyncio.run(demo_improved())