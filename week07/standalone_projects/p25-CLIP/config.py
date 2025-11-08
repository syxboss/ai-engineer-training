"""
配置管理模块
统一管理CLIP图像搜索系统的所有配置参数
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class CLIPConfig:
    """CLIP模型配置"""
    model_name: str = "ViT-B/32"
    feature_dimension: int = 512


@dataclass
class MilvusConfig:
    """Milvus数据库配置"""
    collection_name: str = "image_collection"
    index_name: str = "hnsw_index"
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    index_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.index_params is None:
            self.index_params = {"M": 8, "efConstruction": 64}


@dataclass
class ImageConfig:
    """图像处理配置"""
    image_directory: str = "./reverse_image_search/train"
    image_extensions: tuple = ("*.JPEG", "*.jpg", "*.png", "*.bmp")
    thumbnail_size: tuple = (150, 150)
    grid_columns: int = 5
    grid_rows: int = 2


@dataclass
class SearchConfig:
    """搜索配置"""
    default_limit: int = 10
    output_fields: list = None
    
    def __post_init__(self):
        if self.output_fields is None:
            self.output_fields = ["filepath"]


class Config:
    """主配置类，整合所有配置"""
    
    def __init__(self):
        self.clip = CLIPConfig()
        self.milvus = MilvusConfig()
        self.image = ImageConfig()
        self.search = SearchConfig()
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        # 检查图像目录是否存在
        if not os.path.exists(self.image.image_directory):
            raise ValueError(f"图像目录不存在: {self.image.image_directory}")
        
        # 检查特征维度是否匹配
        if self.clip.feature_dimension != 512:
            raise ValueError("当前CLIP模型特征维度必须为512")
        
        return True
    
    def get_display_config(self) -> Dict[str, int]:
        """获取显示配置"""
        return {
            "width": self.image.thumbnail_size[0] * self.image.grid_columns,
            "height": self.image.thumbnail_size[1] * self.image.grid_rows,
            "thumbnail_width": self.image.thumbnail_size[0],
            "thumbnail_height": self.image.thumbnail_size[1],
            "columns": self.image.grid_columns,
            "rows": self.image.grid_rows
        }


# 全局配置实例
config = Config()