"""
Tavily搜索工具模块
用于获取最新新闻和信息搜索
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from tavily import TavilyClient
from core.logger import app_logger


class TavilySearchTool:
    """Tavily搜索工具类"""
    
    def __init__(self, api_key: str = "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"):
        """
        初始化Tavily搜索工具
        
        Args:
            api_key: Tavily API密钥
        """
        self.api_key = api_key
        self.client = TavilyClient(api_key)
        app_logger.info("Tavily搜索工具初始化完成")
    
    def search_news(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        搜索新闻信息
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            
        Returns:
            包含搜索结果的字典
        """
        try:
            app_logger.info(f"开始搜索新闻: {query}")
            
            # 调用Tavily搜索API
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=True,
                include_raw_content=False
            )
            
            if not response or 'results' not in response:
                return {
                    'success': False,
                    'error': '搜索结果为空',
                    'data': None
                }
            
            # 格式化搜索结果
            formatted_results = []
            for result in response.get('results', []):
                formatted_result = {
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'published_date': result.get('published_date', ''),
                    'score': result.get('score', 0)
                }
                formatted_results.append(formatted_result)
            
            # 构建返回数据
            search_data = {
                'query': query,
                'answer': response.get('answer', ''),
                'results': formatted_results,
                'search_time': datetime.now().isoformat(),
                'total_results': len(formatted_results)
            }
            
            app_logger.info(f"成功获取 {len(formatted_results)} 条搜索结果")
            
            return {
                'success': True,
                'data': search_data,
                'error': None
            }
            
        except Exception as e:
            error_msg = f"搜索失败: {str(e)}"
            app_logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'data': None
            }
    
    def format_search_results(self, search_data: Dict[str, Any]) -> str:
        """
        格式化搜索结果为可读文本
        
        Args:
            search_data: 搜索数据
            
        Returns:
            格式化后的文本
        """
        if not search_data or not search_data.get('results'):
            return "未找到相关搜索结果"
        
        formatted_text = f"🔍 搜索查询: {search_data.get('query', '')}\n\n"
        
        # 添加AI总结（如果有）
        if search_data.get('answer'):
            formatted_text += f"📝 AI总结:\n{search_data['answer']}\n\n"
        
        # 添加搜索结果
        formatted_text += "📰 相关新闻:\n"
        for i, result in enumerate(search_data['results'][:5], 1):
            title = result.get('title', '无标题')
            content = result.get('content', '')
            url = result.get('url', '')
            
            # 截取内容前150个字符
            if len(content) > 150:
                content = content[:150] + "..."
            
            formatted_text += f"\n{i}. {title}\n"
            formatted_text += f"   {content}\n"
            if url:
                formatted_text += f"   🔗 {url}\n"
        
        formatted_text += f"\n⏰ 搜索时间: {search_data.get('search_time', '')}"
        formatted_text += f"\n📊 共找到 {search_data.get('total_results', 0)} 条结果"
        
        return formatted_text
    
    def search_and_format(self, query: str, max_results: int = 5) -> str:
        """
        搜索并格式化结果的便捷方法
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            
        Returns:
            格式化后的搜索结果文本
        """
        search_result = self.search_news(query, max_results)
        
        if not search_result['success']:
            return f"搜索失败: {search_result.get('error', '未知错误')}"
        
        return self.format_search_results(search_result['data'])


# 创建全局实例
tavily_search_tool = TavilySearchTool()