#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于知识图谱的结构化存储系统

这个模块实现了一个具有知识图谱存储能力的AI助手，使用结构化的三元组来存储和检索知识。
主要特性：
- 结构化知识存储（实体-关系-实体）
- 高效的图查询和路径搜索
- 用户隔离的知识图谱管理器
- 持久化存储和索引
- 完整的知识图谱操作API

Author: AI Assistant
Date: 2024
"""

import os
import json
import pickle
import uuid
import atexit
import logging
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
import networkx as nx

# LangChain 相关导入
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ================================
# 配置和常量
# ================================

# 存储配置
STORAGE_CONFIG = {
    'path': "knowledge_graph_storage",
    'data_file': "graph_data.json",
    'index_file': "graph_index.pkl",
    'networkx_file': "networkx_graph.pkl"
}

# 搜索配置
SEARCH_CONFIG = {
    'default_limit': 10,
    'max_path_length': 5,
    'max_display': 5
}

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================================
# 核心数据结构
# ================================

@dataclass
class KnowledgeNode:
    """知识节点类 - 表示知识图谱中的一个实体节点"""
    label: str
    type: str
    user_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class KnowledgeRelation:
    """知识关系类 - 表示知识图谱中的关系边"""
    source_id: str
    target_id: str
    relation_type: str
    user_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class KnowledgeTriple:
    """知识三元组类 - 表示 (主体, 关系, 客体) 三元组"""
    subject: str
    predicate: str
    object: str
    user_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_tuple(self) -> Tuple[str, str, str]:
        """转换为元组格式"""
        return (self.subject, self.predicate, self.object)


# ================================
# 知识图谱存储管理类
# ================================

class KnowledgeGraphManager:
    """知识图谱存储管理器 - 负责管理知识图谱的存储、索引、查询和持久化操作"""
    
    def __init__(self, storage_path: str = STORAGE_CONFIG['path']):
        """初始化知识图谱管理器"""
        self.storage_path = storage_path
        self._setup_paths()
        self._init_data_structures()
        self._initialize_storage()
        
    def _setup_paths(self):
        """设置文件路径"""
        self.paths = {
            'data': os.path.join(self.storage_path, STORAGE_CONFIG['data_file']),
            'index': os.path.join(self.storage_path, STORAGE_CONFIG['index_file']),
            'networkx': os.path.join(self.storage_path, STORAGE_CONFIG['networkx_file'])
        }
    
    def _init_data_structures(self):
        """初始化数据结构"""
        # 核心数据
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.relations: Dict[str, KnowledgeRelation] = {}
        self.triples: List[KnowledgeTriple] = []
        
        # 索引结构 - 使用统一的索引管理
        self.indexes = {
            'node_label': defaultdict(set),
            'node_type': defaultdict(set),
            'relation_type': defaultdict(set),
            'user_nodes': defaultdict(set),
            'user_relations': defaultdict(set)
        }
        
        # NetworkX 图用于高级查询
        self.nx_graph = nx.MultiDiGraph()
        
    def _initialize_storage(self) -> None:
        """初始化存储"""
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            if os.path.exists(self.paths['data']):
                self._load_data()
            else:
                self._create_new_storage()
        except Exception as e:
            logger.error(f"初始化存储失败: {e}")
            self._create_new_storage()
    
    def _create_new_storage(self) -> None:
        """创建新的存储"""
        logger.info("创建新的知识图谱存储")
        print(" [知识图谱] 创建新的存储系统")
        self._rebuild_indexes()
    
    def _load_data(self) -> None:
        """加载现有数据"""
        try:
            # 加载主数据
            with open(self.paths['data'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重建数据结构
            self._rebuild_from_data(data)
            
            # 加载索引
            self._load_indexes()
            
            # 加载NetworkX图
            self._load_networkx_graph()
            
            # 重建索引以确保一致性
            self._rebuild_indexes()
            
            stats = self._get_stats()
            logger.info(f"成功加载知识图谱: {stats['nodes']} 个节点, {stats['relations']} 个关系, {stats['triples']} 个三元组")
            print(f" [知识图谱] 加载现有数据: {stats['nodes']} 节点, {stats['relations']} 关系, {stats['triples']} 三元组")
            
        except Exception as e:
            logger.warning(f"加载数据失败: {e}，将创建新存储")
            self._create_new_storage()
    
    def _rebuild_from_data(self, data: Dict[str, Any]) -> None:
        """从数据重建对象"""
        # 重建节点
        for node_data in data.get('nodes', []):
            node = KnowledgeNode(**node_data)
            self.nodes[node.id] = node
        
        # 重建关系
        for relation_data in data.get('relations', []):
            relation = KnowledgeRelation(**relation_data)
            self.relations[relation.id] = relation
        
        # 重建三元组
        for triple_data in data.get('triples', []):
            triple = KnowledgeTriple(**triple_data)
            self.triples.append(triple)
    
    def _load_indexes(self) -> None:
        """加载索引"""
        if os.path.exists(self.paths['index']):
            with open(self.paths['index'], 'rb') as f:
                index_data = pickle.load(f)
                for key, value in index_data.items():
                    if key in self.indexes:
                        self.indexes[key] = defaultdict(set, value)
    
    def _load_networkx_graph(self) -> None:
        """加载NetworkX图"""
        if os.path.exists(self.paths['networkx']):
            with open(self.paths['networkx'], 'rb') as f:
                self.nx_graph = pickle.load(f)
    
    def _rebuild_indexes(self) -> None:
        """重建所有索引"""
        # 清空索引
        for index in self.indexes.values():
            index.clear()
        self.nx_graph.clear()
        
        # 重建节点索引
        for node_id, node in self.nodes.items():
            self._add_node_to_indexes(node_id, node)
        
        # 重建关系索引
        for relation_id, relation in self.relations.items():
            self._add_relation_to_indexes(relation_id, relation)
    
    def _add_node_to_indexes(self, node_id: str, node: KnowledgeNode) -> None:
        """将节点添加到索引"""
        self.indexes['node_label'][node.label.lower()].add(node_id)
        self.indexes['node_type'][node.type].add(node_id)
        self.indexes['user_nodes'][node.user_id].add(node_id)
        self.nx_graph.add_node(node_id, **asdict(node))
    
    def _add_relation_to_indexes(self, relation_id: str, relation: KnowledgeRelation) -> None:
        """将关系添加到索引"""
        self.indexes['relation_type'][relation.relation_type].add(relation_id)
        self.indexes['user_relations'][relation.user_id].add(relation_id)
        self.nx_graph.add_edge(
            relation.source_id,
            relation.target_id,
            key=relation_id,
            **asdict(relation)
        )
    
    def _get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            'nodes': len(self.nodes),
            'relations': len(self.relations),
            'triples': len(self.triples)
        }
    
    def save_data(self) -> bool:
        """保存数据到磁盘"""
        try:
            stats = self._get_stats()
            
            # 保存主数据
            data = {
                'nodes': [asdict(node) for node in self.nodes.values()],
                'relations': [asdict(relation) for relation in self.relations.values()],
                'triples': [asdict(triple) for triple in self.triples],
                'metadata': {
                    'version': '1.0',
                    'timestamp': datetime.now().isoformat(),
                    **stats
                }
            }
            
            with open(self.paths['data'], 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 保存索引
            index_data = {key: dict(value) for key, value in self.indexes.items()}
            with open(self.paths['index'], 'wb') as f:
                pickle.dump(index_data, f)
            
            # 保存NetworkX图
            with open(self.paths['networkx'], 'wb') as f:
                pickle.dump(self.nx_graph, f)
            
            logger.info(f"知识图谱已保存到 {self.storage_path}")
            print(f" [知识图谱] 数据已保存 ({stats['nodes']} 节点, {stats['relations']} 关系)")
            return True
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            print(f" [知识图谱] 保存失败: {e}")
            return False
    
    def add_node(self, label: str, node_type: str, properties: Dict[str, Any], user_id: str) -> str:
        """添加知识节点"""
        try:
            node = KnowledgeNode(
                label=label,
                type=node_type,
                properties=properties,
                user_id=user_id
            )
            
            self.nodes[node.id] = node
            self._add_node_to_indexes(node.id, node)
            
            logger.info(f"添加节点成功: {label} ({node_type})")
            return node.id
            
        except Exception as e:
            logger.error(f"添加节点失败: {e}")
            return ""
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str, 
                    properties: Dict[str, Any], user_id: str, weight: float = 1.0) -> str:
        """添加知识关系"""
        try:
            # 检查节点是否存在
            if source_id not in self.nodes or target_id not in self.nodes:
                logger.error("源节点或目标节点不存在")
                return ""
            
            relation = KnowledgeRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                properties=properties,
                user_id=user_id,
                weight=weight
            )
            
            self.relations[relation.id] = relation
            self._add_relation_to_indexes(relation.id, relation)
            
            logger.info(f"添加关系成功: {relation_type} ({source_id} -> {target_id})")
            return relation.id
            
        except Exception as e:
            logger.error(f"添加关系失败: {e}")
            return ""
    
    def add_triple(self, subject: str, predicate: str, obj: str, 
                  properties: Dict[str, Any], user_id: str, confidence: float = 1.0) -> bool:
        """添加知识三元组"""
        try:
            triple = KnowledgeTriple(
                subject=subject,
                predicate=predicate,
                object=obj,
                properties=properties,
                user_id=user_id,
                confidence=confidence
            )
            
            self.triples.append(triple)
            
            # 自动创建节点和关系
            subject_id = self._ensure_node_exists(subject, "entity", user_id)
            object_id = self._ensure_node_exists(obj, "entity", user_id)
            
            if subject_id and object_id:
                self.add_relation(subject_id, object_id, predicate, properties, user_id)
            
            logger.info(f"添加三元组成功: ({subject}, {predicate}, {obj})")
            return True
            
        except Exception as e:
            logger.error(f"添加三元组失败: {e}")
            return False
    
    def _ensure_node_exists(self, label: str, node_type: str, user_id: str) -> str:
        """确保节点存在，如果不存在则创建"""
        existing_nodes = self.search_nodes_by_label(label, user_id)
        return existing_nodes[0].id if existing_nodes else self.add_node(label, node_type, {}, user_id)
    
    # ================================
    # 查询和搜索方法
    # ================================
    
    def _search_nodes_by_index(self, index_key: str, search_value: str, user_id: str) -> List[KnowledgeNode]:
        """通用节点搜索方法"""
        try:
            node_ids = self.indexes[index_key].get(search_value, set())
            user_node_ids = self.indexes['user_nodes'].get(user_id, set())
            matching_ids = node_ids.intersection(user_node_ids)
            return [self.nodes[node_id] for node_id in matching_ids if node_id in self.nodes]
        except Exception as e:
            logger.error(f"搜索节点失败: {e}")
            return []
    
    def search_nodes_by_label(self, label: str, user_id: str) -> List[KnowledgeNode]:
        """根据标签搜索节点"""
        return self._search_nodes_by_index('node_label', label.lower(), user_id)
    
    def search_nodes_by_type(self, node_type: str, user_id: str) -> List[KnowledgeNode]:
        """根据类型搜索节点"""
        return self._search_nodes_by_index('node_type', node_type, user_id)
    
    def search_relations_by_type(self, relation_type: str, user_id: str) -> List[KnowledgeRelation]:
        """根据类型搜索关系"""
        try:
            relation_ids = self.indexes['relation_type'].get(relation_type, set())
            user_relation_ids = self.indexes['user_relations'].get(user_id, set())
            matching_ids = relation_ids.intersection(user_relation_ids)
            return [self.relations[relation_id] for relation_id in matching_ids if relation_id in self.relations]
        except Exception as e:
            logger.error(f"按类型搜索关系失败: {e}")
            return []
    
    def get_node_relations(self, node_id: str, user_id: str, direction: str = "both") -> List[KnowledgeRelation]:
        """获取节点的所有关系"""
        try:
            relations = []
            user_relation_ids = self.indexes['user_relations'].get(user_id, set())
            
            for relation_id in user_relation_ids:
                if relation_id not in self.relations:
                    continue
                    
                relation = self.relations[relation_id]
                
                if direction == "out" and relation.source_id == node_id:
                    relations.append(relation)
                elif direction == "in" and relation.target_id == node_id:
                    relations.append(relation)
                elif direction == "both" and (relation.source_id == node_id or relation.target_id == node_id):
                    relations.append(relation)
            
            return relations
            
        except Exception as e:
            logger.error(f"获取节点关系失败: {e}")
            return []
    
    def find_shortest_path(self, source_id: str, target_id: str, user_id: str) -> List[str]:
        """查找两个节点之间的最短路径"""
        try:
            user_graph = self._create_user_subgraph(user_id)
            
            if source_id not in user_graph or target_id not in user_graph:
                return []
            
            try:
                return nx.shortest_path(user_graph, source_id, target_id)
            except nx.NetworkXNoPath:
                return []
                
        except Exception as e:
            logger.error(f"查找最短路径失败: {e}")
            return []
    
    def find_paths_with_relation(self, source_id: str, relation_type: str, user_id: str, 
                               max_length: int = SEARCH_CONFIG['max_path_length']) -> List[Tuple[str, str]]:
        """查找通过特定关系类型连接的路径"""
        try:
            paths = []
            visited = set()
            queue = deque([(source_id, [source_id], 0)])
            
            while queue:
                current_node, path, length = queue.popleft()
                
                if length >= max_length or current_node in visited:
                    continue
                    
                visited.add(current_node)
                
                # 获取当前节点的出边关系
                relations = self.get_node_relations(current_node, user_id, "out")
                
                for relation in relations:
                    if relation.relation_type == relation_type:
                        target_id = relation.target_id
                        new_path = path + [target_id]
                        
                        if target_id != source_id:  # 避免回到起点
                            path_desc = " -> ".join([
                                self.nodes.get(nid, KnowledgeNode("", f"Node_{nid}", "", {}, "", "")).label 
                                for nid in new_path
                            ])
                            paths.append((target_id, path_desc))
                            
                            if length + 1 < max_length:
                                queue.append((target_id, new_path, length + 1))
            
            return paths[:SEARCH_CONFIG['default_limit']]
            
        except Exception as e:
            logger.error(f"查找关系路径失败: {e}")
            return []
    
    def search_triples(self, subject: str = None, predicate: str = None, obj: str = None, 
                      user_id: str = None) -> List[KnowledgeTriple]:
        """搜索三元组"""
        try:
            results = []
            
            for triple in self.triples:
                # 检查用户ID
                if user_id and triple.user_id != user_id:
                    continue
                
                # 检查主体、谓词、客体
                if (subject and triple.subject.lower() != subject.lower()) or \
                   (predicate and triple.predicate.lower() != predicate.lower()) or \
                   (obj and triple.object.lower() != obj.lower()):
                    continue
                
                results.append(triple)
            
            return results[:SEARCH_CONFIG['default_limit']]
            
        except Exception as e:
            logger.error(f"搜索三元组失败: {e}")
            return []
    
    def _create_user_subgraph(self, user_id: str) -> nx.MultiDiGraph:
        """创建用户的子图"""
        try:
            user_graph = nx.MultiDiGraph()
            
            # 添加用户的节点
            user_node_ids = self.indexes['user_nodes'].get(user_id, set())
            for node_id in user_node_ids:
                if node_id in self.nodes:
                    user_graph.add_node(node_id, **asdict(self.nodes[node_id]))
            
            # 添加用户的关系
            user_relation_ids = self.indexes['user_relations'].get(user_id, set())
            for relation_id in user_relation_ids:
                if relation_id in self.relations:
                    relation = self.relations[relation_id]
                    if relation.source_id in user_graph and relation.target_id in user_graph:
                        user_graph.add_edge(
                            relation.source_id,
                            relation.target_id,
                            key=relation_id,
                            **asdict(relation)
                        )
            
            return user_graph
            
        except Exception as e:
            logger.error(f"创建用户子图失败: {e}")
            return nx.MultiDiGraph()
    
    def get_stats(self, user_id: str = None) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            if user_id:
                user_nodes = len(self.indexes['user_nodes'].get(user_id, set()))
                user_relations = len(self.indexes['user_relations'].get(user_id, set()))
                user_triples = len([t for t in self.triples if t.user_id == user_id])
                
                return {
                    "user_id": user_id,
                    "nodes": user_nodes,
                    "relations": user_relations,
                    "triples": user_triples,
                    "storage_path": self.storage_path
                }
            else:
                return {
                    "total_nodes": len(self.nodes),
                    "total_relations": len(self.relations),
                    "total_triples": len(self.triples),
                    "total_users": len(self.indexes['user_nodes']),
                    "storage_path": self.storage_path,
                    "node_types": list(self.indexes['node_type'].keys()),
                    "relation_types": list(self.indexes['relation_type'].keys())
                }
                
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


# ================================
# 全局实例和配置
# ================================

# 初始化知识图谱管理器
knowledge_manager = KnowledgeGraphManager()

# 初始化模型
model = ChatTongyi(
    model="qwen-max",
    temperature=0.7,
    streaming=True
)

# 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个具有知识图谱存储能力的AI助手。

知识图谱能力说明：
- 我可以存储和管理结构化的知识信息
- 我可以理解实体之间的关系和连接
- 我可以进行复杂的知识查询和推理
- 如果需要保存知识，我会使用知识图谱工具

当前相关知识：
{knowledge_context}

请根据上述知识和当前对话，提供有帮助的回答。"""),
    MessagesPlaceholder(variable_name="messages")
])


# ================================
# 工具函数
# ================================

def get_user_id(config: RunnableConfig) -> str:
    """从配置中提取用户ID"""
    return config["configurable"].get("user_id", "default_user")


@tool
def add_knowledge_node(label: str, node_type: str, properties: str, config: RunnableConfig) -> str:
    """添加知识节点到图谱"""
    user_id = get_user_id(config)
    
    try:
        props = json.loads(properties) if properties else {}
    except:
        props = {}
    
    node_id = knowledge_manager.add_node(label, node_type, props, user_id)
    if node_id:
        knowledge_manager.save_data()
        return f"✓ 知识节点已添加: {label} ({node_type})"
    else:
        return "✗ 知识节点添加失败"


@tool
def add_knowledge_relation(source_label: str, target_label: str, relation_type: str, properties: str, config: RunnableConfig) -> str:
    """添加知识关系到图谱"""
    user_id = get_user_id(config)
    
    try:
        props = json.loads(properties) if properties else {}
    except:
        props = {}
    
    # 查找或创建节点
    source_nodes = knowledge_manager.search_nodes_by_label(source_label, user_id)
    target_nodes = knowledge_manager.search_nodes_by_label(target_label, user_id)
    
    source_id = source_nodes[0].id if source_nodes else knowledge_manager.add_node(source_label, "entity", {}, user_id)
    target_id = target_nodes[0].id if target_nodes else knowledge_manager.add_node(target_label, "entity", {}, user_id)
    
    if source_id and target_id:
        relation_id = knowledge_manager.add_relation(source_id, target_id, relation_type, props, user_id)
        if relation_id:
            knowledge_manager.save_data()
            return f"✓ 知识关系已添加: {source_label} --{relation_type}--> {target_label}"
        else:
            return "✗ 知识关系添加失败"
    else:
        return "✗ 无法创建或找到相关节点"


@tool
def add_knowledge_triple(subject: str, predicate: str, obj: str, properties: str, config: RunnableConfig) -> str:
    """添加知识三元组到图谱"""
    user_id = get_user_id(config)
    
    try:
        props = json.loads(properties) if properties else {}
    except:
        props = {}
    
    success = knowledge_manager.add_triple(subject, predicate, obj, props, user_id)
    if success:
        knowledge_manager.save_data()
        return f"✓ 知识三元组已添加: ({subject}, {predicate}, {obj})"
    else:
        return "✗ 知识三元组添加失败"


@tool
def search_knowledge_nodes(query: str, search_type: str, config: RunnableConfig) -> List[str]:
    """搜索知识节点"""
    user_id = get_user_id(config)
    
    try:
        if search_type == "label":
            nodes = knowledge_manager.search_nodes_by_label(query, user_id)
        elif search_type == "type":
            nodes = knowledge_manager.search_nodes_by_type(query, user_id)
        else:
            return ["搜索类型无效，请使用 'label' 或 'type'"]
        
        if nodes:
            results = []
            for node in nodes[:SEARCH_CONFIG['max_display']]:
                result = f"节点: {node.label} (类型: {node.type})"
                if node.properties:
                    result += f" - 属性: {node.properties}"
                results.append(result)
            
            print(f" [知识检索] 找到 {len(nodes)} 个节点")
            return results
        else:
            return ["未找到匹配的知识节点"]
            
    except Exception as e:
        logger.error(f"搜索知识节点失败: {e}")
        return [f"搜索失败: {e}"]


@tool
def search_knowledge_relations(node_label: str, direction: str, config: RunnableConfig) -> List[str]:
    """搜索知识关系"""
    user_id = get_user_id(config)
    
    try:
        # 查找节点
        nodes = knowledge_manager.search_nodes_by_label(node_label, user_id)
        if not nodes:
            return [f"未找到节点: {node_label}"]
        
        node = nodes[0]
        relations = knowledge_manager.get_node_relations(node.id, user_id, direction)
        
        if relations:
            results = []
            for relation in relations[:SEARCH_CONFIG['max_display']]:
                source_node = knowledge_manager.nodes.get(relation.source_id)
                target_node = knowledge_manager.nodes.get(relation.target_id)
                
                if source_node and target_node:
                    result = f"{source_node.label} --{relation.relation_type}--> {target_node.label}"
                    if relation.properties:
                        result += f" (属性: {relation.properties})"
                    results.append(result)
            
            print(f" [关系检索] 找到 {len(relations)} 个关系")
            return results
        else:
            return [f"节点 {node_label} 没有找到相关关系"]
            
    except Exception as e:
        logger.error(f"搜索知识关系失败: {e}")
        return [f"搜索失败: {e}"]


@tool
def search_knowledge_path(source_label: str, target_label: str, config: RunnableConfig) -> List[str]:
    """查找节点间的路径"""
    user_id = get_user_id(config)
    
    try:
        # 查找节点
        source_nodes = knowledge_manager.search_nodes_by_label(source_label, user_id)
        target_nodes = knowledge_manager.search_nodes_by_label(target_label, user_id)
        
        if not source_nodes:
            return [f"未找到源节点: {source_label}"]
        if not target_nodes:
            return [f"未找到目标节点: {target_label}"]
        
        source_id = source_nodes[0].id
        target_id = target_nodes[0].id
        
        path = knowledge_manager.find_shortest_path(source_id, target_id, user_id)
        
        if path:
            # 构建路径描述
            path_labels = []
            for node_id in path:
                node = knowledge_manager.nodes.get(node_id)
                if node:
                    path_labels.append(node.label)
            
            path_desc = " -> ".join(path_labels)
            print(f" [路径检索] 找到路径: {path_desc}")
            return [f"最短路径: {path_desc}"]
        else:
            return [f"未找到从 {source_label} 到 {target_label} 的路径"]
            
    except Exception as e:
        logger.error(f"搜索知识路径失败: {e}")
        return [f"搜索失败: {e}"]


# ================================
# 状态管理
# ================================

class State(MessagesState):
    """对话状态类 - 扩展MessagesState以包含知识图谱相关字段"""
    knowledge_context: List[str]


# ================================
# 节点函数
# ================================

def load_knowledge(state: State, config: RunnableConfig) -> State:
    """加载相关知识节点"""
    # 获取对话内容用于搜索
    convo_str = get_buffer_string(state["messages"])
    user_id = get_user_id(config)
    
    # 搜索相关知识
    knowledge_context = []
    
    # 尝试从对话中提取实体和关系进行搜索
    words = convo_str.lower().split()
    
    # 搜索节点
    for word in words:
        if len(word) > 2:  # 过滤短词
            nodes = knowledge_manager.search_nodes_by_label(word, user_id)
            for node in nodes[:2]:  # 限制数量
                context = f"实体: {node.label} (类型: {node.type})"
                if node.properties:
                    context += f" - {node.properties}"
                knowledge_context.append(context)
    
    # 搜索三元组
    for word in words:
        if len(word) > 2:
            triples = knowledge_manager.search_triples(subject=word, user_id=user_id)
            for triple in triples[:2]:
                context = f"关系: {triple.subject} --{triple.predicate}--> {triple.object}"
                knowledge_context.append(context)
    
    return {"knowledge_context": knowledge_context[:SEARCH_CONFIG['max_display']]}


def agent(state: State) -> State:
    """AI代理节点"""
    # 绑定工具到模型
    model_with_tools = model.bind_tools([
        add_knowledge_node,
        add_knowledge_relation,
        add_knowledge_triple,
        search_knowledge_nodes,
        search_knowledge_relations,
        search_knowledge_path
    ])
    
    # 准备知识上下文字符串
    knowledge_str = ""
    if state.get("knowledge_context"):
        knowledge_str = "\n".join([f"• {knowledge}" for knowledge in state["knowledge_context"]])
    
    # 生成回复
    response = prompt.invoke({
        "messages": state["messages"],
        "knowledge_context": knowledge_str or "暂无相关知识"
    })
    
    prediction = model_with_tools.invoke(response)
    
    return {"messages": state["messages"] + [prediction]}


def save_knowledge(state: State, config: RunnableConfig) -> State:
    """保存知识节点"""
    messages = state["messages"]
    
    if len(messages) >= 2:
        user_msg = None
        ai_msg = None
        
        # 提取最近的用户消息和AI回复
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
        
        # 自动提取和保存知识（简单的实体识别）
        if user_msg:
            user_id = get_user_id(config)
            # 这里可以添加更复杂的NLP处理来提取实体和关系
            # 目前只是一个简单的演示
            pass
    
    return {}


def route_tools(state: State):
    """工具路由节点"""
    msg = state["messages"][-1]
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        return "tools"
    return END


# ================================
# 图构建
# ================================

def build_graph() -> StateGraph:
    """构建状态图"""
    builder = StateGraph(State)
    
    # 添加节点
    builder.add_node("load_knowledge", load_knowledge)
    builder.add_node("agent", agent)
    builder.add_node("save_knowledge", save_knowledge)
    builder.add_node("tools", ToolNode([
        add_knowledge_node,
        add_knowledge_relation,
        add_knowledge_triple,
        search_knowledge_nodes,
        search_knowledge_relations,
        search_knowledge_path
    ]))
    
    # 添加边
    builder.add_edge(START, "load_knowledge")
    builder.add_edge("load_knowledge", "agent")
    builder.add_edge("agent", "save_knowledge")
    builder.add_conditional_edges("save_knowledge", route_tools, ["tools", END])
    builder.add_edge("tools", "agent")
    
    # 编译图
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ================================
# 演示和工具函数
# ================================

def print_separator(title: str = "", char: str = "=", width: int = 60) -> None:
    """打印分隔符"""
    if title:
        title_line = f" {title} "
        padding = (width - len(title_line)) // 2
        print(f"{char * padding}{title_line}{char * padding}")
    else:
        print(char * width)


def print_stats(user_id: str = None) -> None:
    """打印统计信息"""
    stats = knowledge_manager.get_stats(user_id)
    
    if user_id:
        print(f"用户 {user_id} 的知识图谱统计")
        print(f" 节点数量: {stats.get('nodes', 0)}")
        print(f" 关系数量: {stats.get('relations', 0)}")
        print(f" 三元组数量: {stats.get('triples', 0)}")
    else:
        print("知识图谱系统统计")
        print(f" 总节点数量: {stats.get('total_nodes', 0)}")
        print(f" 总关系数量: {stats.get('total_relations', 0)}")
        print(f" 总三元组数量: {stats.get('total_triples', 0)}")
        print(f" 用户数量: {stats.get('total_users', 0)}")
        print(f" 存储路径: {stats.get('storage_path', 'N/A')}")
        print(f" 节点类型: {', '.join(stats.get('node_types', []))}")
        print(f" 关系类型: {', '.join(stats.get('relation_types', []))}")


def get_stream_chunk(chunk: Dict[str, Any]) -> None:
    """处理流式输出块"""
    for node, update in chunk.items():
        if update is None:
            continue
        
        # 处理消息输出
        msgs = update.get("messages")
        if msgs:
            last = msgs[-1]
            content = getattr(last, "content", None) or (
                last.get("content") if isinstance(last, dict) else None
            )
            if content:
                print(f" 助手: {content}")
        
        # 处理知识输出
        if "knowledge_context" in update and update["knowledge_context"]:
            knowledge = update["knowledge_context"]
            print(f" [激活知识] 检索到 {len(knowledge)} 条相关知识")


def run_demo_conversation(graph: StateGraph, config: Dict[str, Any]) -> None:
    """运行演示对话"""
    demo_conversations = [
        {
            "title": "第一轮对话 - 建立知识图谱",
            "message": "苹果是一种水果，它含有丰富的维生素C"
        },
        {
            "title": "第二轮对话 - 添加更多知识",
            "message": "香蕉也是水果，它富含钾元素。苹果和香蕉都属于健康食品"
        },
        {
            "title": "第三轮对话 - 知识查询",
            "message": "告诉我关于苹果的信息"
        },
        {
            "title": "第四轮对话 - 关系查询",
            "message": "水果类食物有哪些？它们之间有什么关系？"
        },
        {
            "title": "第五轮对话 - 路径查询",
            "message": "苹果和健康食品之间有什么联系？"
        }
    ]
    
    for i, conv in enumerate(demo_conversations, 1):
        print(f"{conv['title']}")
        print(f" 用户: {conv['message']}")
        print()
        
        for chunk in graph.stream(
            {"messages": [{"role": "user", "content": conv["message"]}]},
            config=config
        ):
            get_stream_chunk(chunk)
        
        print()
        if i < len(demo_conversations):
            input("按回车键继续下一轮对话...")
            print()


def interactive_mode(graph: StateGraph, config: Dict[str, Any]) -> None:
    """交互模式"""
    print("进入交互模式 (输入 'quit' 退出, 'stats' 查看统计, 'help' 查看帮助)")
    print()
    
    while True:
        try:
            user_input = input(" 用户: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stats':
                print_stats(config["configurable"]["user_id"])
                continue
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif not user_input:
                continue
            
            print()
            for chunk in graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config
            ):
                get_stream_chunk(chunk)
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f" 错误: {e}")


def print_help() -> None:
    """打印帮助信息"""
    print("知识图谱助手帮助")
    print(" 可用命令:")
    print("   quit  - 退出程序")
    print("   stats - 查看统计信息")
    print("   help  - 显示此帮助")
    print()
    print(" 知识图谱功能:")
    print("   - 自动识别和存储实体、关系")
    print("   - 支持复杂的知识查询和推理")
    print("   - 可以查找实体间的路径和连接")
    print("   - 持久化存储，重启后数据不丢失")


# ================================
# 主程序
# ================================

def main():
    """主程序入口"""
    print("知识图谱结构化存储系统演示")
    print(" 本演示将展示基于知识图谱的结构化存储和查询能力")
    print(" 特性: 实体关系存储、图查询、路径搜索、持久化存储")
    print()
    
    # 显示初始统计
    print_stats()
    
    # 构建图
    graph = build_graph()
    
    # 配置
    config = {
        "configurable": {
            "thread_id": "knowledge-demo-thread-1",
            "user_id": "knowledge-demo-user-1",
        }
    }
    
    # 选择运行模式
    print("\n请选择运行模式:")
    print("1. 演示模式 (预设对话)")
    print("2. 交互模式 (自由对话)")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        print()
        
        if choice == "1":
            run_demo_conversation(graph, config)
        elif choice == "2":
            interactive_mode(graph, config)
        else:
            print("无效选择，使用演示模式")
            run_demo_conversation(graph, config)
        
        print("演示完成")
        print_stats(config["configurable"]["user_id"])
        print(" 知识图谱系统演示完成！")
        print(" 知识已持久化保存，重启程序后仍可使用")
        
    except KeyboardInterrupt:
        print("\n\n  演示被用户中断")
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        print(f" 演示失败: {e}")
    finally:
        # 确保保存数据
        knowledge_manager.save_data()


# 注册退出时保存数据
atexit.register(knowledge_manager.save_data)


if __name__ == "__main__":
    main()