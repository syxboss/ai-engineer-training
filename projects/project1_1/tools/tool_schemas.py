"""
工具模式定义
定义用于LangChain工具调用的Pydantic模型
"""

from pydantic.v1 import BaseModel, Field
from typing import Optional


class WeatherQuery(BaseModel):
    """天气查询工具"""
    
    city_name: str = Field(
        ..., 
        description="要查询天气的城市名称，例如：北京、上海、广州等"
    )
    
    class Config:
        schema_extra = {
            "examples": [
                {"city_name": "北京"},
                {"city_name": "上海"},
                {"city_name": "广州"}
            ]
        }


class NewsSearch(BaseModel):
    """新闻搜索工具"""
    
    query: str = Field(
        ..., 
        description="搜索关键词或问题，用于查找相关新闻和信息"
    )
    
    max_results: Optional[int] = Field(
        default=5,
        description="返回的最大结果数量，默认为5"
    )
    
    class Config:
        schema_extra = {
            "examples": [
                {"query": "人工智能最新发展", "max_results": 5},
                {"query": "科技新闻", "max_results": 3},
                {"query": "财经资讯"}
            ]
        }