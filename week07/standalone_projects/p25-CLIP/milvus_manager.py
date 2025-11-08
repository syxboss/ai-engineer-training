"""
Milvus数据库管理模块
封装Milvus向量数据库的所有操作，包括集合管理、数据插入和搜索
"""

import logging
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient
from config import config


class MilvusManager:
    """Milvus向量数据库管理器
    
    提供集合创建、索引管理、数据插入和向量搜索等功能
    """
    
    def __init__(self, uri: str = None):
        """初始化Milvus管理器
        
        Args:
            uri: Milvus服务器URI，默认使用本地Lite模式
        """
        self.uri = uri
        self.client = None
        self.collection_name = config.milvus.collection_name
        self._connect()
    
    def _connect(self) -> None:
        """连接到Milvus服务器"""
        try:
            if self.uri:
                self.client = MilvusClient(uri=self.uri)
                logging.info(f"已连接到Milvus服务器: {self.uri}")
            else:
                self.client = MilvusClient()
                logging.info("已连接到Milvus Lite本地模式")
        except Exception as e:
            logging.error(f"Milvus连接失败: {e}")
            raise
    
    def create_collection(self, 
                         collection_name: str = None,
                         dimension: int = None,
                         drop_existing: bool = True) -> bool:
        """创建向量集合
        
        Args:
            collection_name: 集合名称，默认使用配置中的名称
            dimension: 向量维度，默认使用配置中的维度
            drop_existing: 是否删除已存在的同名集合
            
        Returns:
            创建是否成功
        """
        collection_name = collection_name or self.collection_name
        dimension = dimension or config.clip.feature_dimension
        
        try:
            # 检查集合是否存在
            if self.client.has_collection(collection_name):
                if drop_existing:
                    logging.info(f"删除已存在的集合: {collection_name}")
                    self.client.drop_collection(collection_name)
                else:
                    logging.info(f"集合已存在: {collection_name}")
                    return True
            
            # 创建集合
            self.client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                auto_id=True,
                enable_dynamic_field=True
            )
            
            logging.info(f"成功创建集合: {collection_name}, 维度: {dimension}")
            return True
            
        except Exception as e:
            logging.error(f"创建集合失败: {e}")
            return False
    
    def create_index(self, 
                    collection_name: str = None,
                    field_name: str = None) -> bool:
        """创建向量索引
        
        Args:
            collection_name: 集合名称
            field_name: 字段名称，默认为'vector'
            
        Returns:
            索引创建是否成功
        """
        collection_name = collection_name or self.collection_name
        field_name = field_name or "vector"
        
        try:
            # 检查索引是否已存在
            existing_indexes = self.client.list_indexes(collection_name=collection_name)
            if field_name in existing_indexes:
                logging.info(f"索引已存在，字段: {field_name}")
                return True
            
            # 使用新的API创建IndexParams对象
            index_params = self.client.prepare_index_params()
            
            # 添加向量字段的索引
            index_params.add_index(
                field_name=field_name,
                index_type=config.milvus.index_type,
                metric_type=config.milvus.metric_type,
                params=config.milvus.index_params
            )
            
            # 创建索引
            self.client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )
            
            logging.info(f"成功创建索引，字段: {field_name}")
            return True
            
        except Exception as e:
            logging.error(f"创建索引失败: {e}")
            return False
    
    def insert_data(self, 
                   data: List[Dict[str, Any]], 
                   collection_name: str = None) -> Optional[Dict[str, Any]]:
        """插入向量数据
        
        Args:
            data: 要插入的数据列表，每个元素包含vector和其他字段
            collection_name: 集合名称
            
        Returns:
            插入结果信息，失败时返回None
        """
        collection_name = collection_name or self.collection_name
        
        try:
            result = self.client.insert(
                collection_name=collection_name,
                data=data
            )
            
            insert_count = result.get('insert_count', 0)
            logging.info(f"成功插入 {insert_count} 条数据到集合 {collection_name}")
            return result
            
        except Exception as e:
            logging.error(f"数据插入失败: {e}")
            return None
    
    def search(self, 
              query_vectors: List[List[float]], 
              collection_name: str = None,
              limit: int = None,
              output_fields: List[str] = None,
              search_params: Dict[str, Any] = None) -> Optional[List[List[Dict]]]:
        """向量相似性搜索
        
        Args:
            query_vectors: 查询向量列表
            collection_name: 集合名称
            limit: 返回结果数量限制
            output_fields: 需要返回的字段列表
            search_params: 搜索参数
            
        Returns:
            搜索结果列表，失败时返回None
        """
        collection_name = collection_name or self.collection_name
        limit = limit or config.search.default_limit
        output_fields = output_fields or config.search.output_fields
        
        if search_params is None:
            search_params = {"metric_type": config.milvus.metric_type}
        
        try:
            results = self.client.search(
                collection_name=collection_name,
                data=query_vectors,
                limit=limit,
                output_fields=output_fields,
                search_params=search_params
            )
            
            logging.info(f"搜索完成，返回 {len(results)} 组结果")
            return results
            
        except Exception as e:
            logging.error(f"向量搜索失败: {e}")
            return None
    
    def get_collection_stats(self, collection_name: str = None) -> Optional[Dict[str, Any]]:
        """获取集合统计信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合统计信息字典
        """
        collection_name = collection_name or self.collection_name
        
        try:
            if not self.client.has_collection(collection_name):
                logging.warning(f"集合不存在: {collection_name}")
                return None
            
            # 获取集合信息
            stats = self.client.get_collection_stats(collection_name)
            return stats
            
        except Exception as e:
            logging.error(f"获取集合统计信息失败: {e}")
            return None
    
    def delete_collection(self, collection_name: str = None) -> bool:
        """删除集合
        
        Args:
            collection_name: 集合名称
            
        Returns:
            删除是否成功
        """
        collection_name = collection_name or self.collection_name
        
        try:
            if self.client.has_collection(collection_name):
                self.client.drop_collection(collection_name)
                logging.info(f"成功删除集合: {collection_name}")
                return True
            else:
                logging.warning(f"集合不存在: {collection_name}")
                return False
                
        except Exception as e:
            logging.error(f"删除集合失败: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """列出所有集合
        
        Returns:
            集合名称列表
        """
        try:
            collections = self.client.list_collections()
            logging.info(f"找到 {len(collections)} 个集合")
            return collections
        except Exception as e:
            logging.error(f"列出集合失败: {e}")
            return []
    
    def check_image_exists(self, 
                          filepath: str, 
                          collection_name: str = None) -> bool:
        """检查图片是否已存在于数据库中
        
        Args:
            filepath: 图片文件路径
            collection_name: 集合名称
            
        Returns:
            图片是否已存在
        """
        collection_name = collection_name or self.collection_name
        
        try:
            # 检查集合是否存在
            if not self.client.has_collection(collection_name):
                return False
            
            # 使用filepath字段查询
            results = self.client.query(
                collection_name=collection_name,
                filter=f'filepath == "{filepath}"',
                output_fields=["filepath"],
                limit=1
            )
            
            return len(results) > 0
            
        except Exception as e:
            logging.error(f"检查图片是否存在失败: {e}")
            return False
    
    def batch_check_images_exist(self, 
                                filepaths: List[str], 
                                collection_name: str = None) -> Dict[str, bool]:
        """批量检查图片是否已存在于数据库中
        
        Args:
            filepaths: 图片文件路径列表
            collection_name: 集合名称
            
        Returns:
            文件路径到存在状态的映射字典
        """
        collection_name = collection_name or self.collection_name
        result = {}
        
        try:
            # 检查集合是否存在
            if not self.client.has_collection(collection_name):
                return {path: False for path in filepaths}
            
            # 批量查询所有已存在的文件路径
            if filepaths:
                # 构建查询条件
                filepath_conditions = [f'filepath == "{path}"' for path in filepaths]
                filter_expr = " or ".join(filepath_conditions)
                
                existing_results = self.client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=["filepath"],
                    limit=len(filepaths)
                )
                
                # 提取已存在的文件路径
                existing_paths = {result["filepath"] for result in existing_results}
                
                # 构建结果字典
                for path in filepaths:
                    result[path] = path in existing_paths
            
            return result
            
        except Exception as e:
            logging.error(f"批量检查图片是否存在失败: {e}")
            return {path: False for path in filepaths}

    def close(self) -> None:
        """关闭连接"""
        if self.client:
            # Milvus客户端通常不需要显式关闭
            logging.info("Milvus连接已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()