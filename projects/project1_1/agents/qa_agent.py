"""
问答智能代理模块
基于 LangChain 管道语法构建的简化问答助手
使用 prompt | llm | output 模式
"""

import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

# 禁用 LangChain 追踪
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
import uuid
import time
from typing import Dict, Any, List

from config.settings import settings
from core.logger import app_logger
from tools.amap_weather_tool import AmapWeatherTool
from tools.tavily_search_tool import TavilySearchTool
from tools.tool_schemas import WeatherQuery, NewsSearch


class QAAgent:
    """简化的问答代理，使用 LangChain 的 prompt | llm | output 语法"""
    
    def __init__(self, session_id: str = None):
        """初始化问答代理"""
        self.session_id = session_id or str(uuid.uuid4())
        self.conversation_history = []
        self.weather_tool = AmapWeatherTool()
        self.search_tool = TavilySearchTool()
        
        # 创建工具函数
        self.tools = self._create_tools()
        
        # 初始化LLM并绑定工具
        self.llm = self._initialize_llm()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # 创建通用对话链
        self.general_chain = self._create_general_chain()
        
        app_logger.info(f"QA代理初始化完成，会话ID: {self.session_id}")
    
    def _create_tools(self):
        """创建工具函数列表"""
        
        @tool("weather_query", args_schema=WeatherQuery)
        def weather_query(city_name: str) -> str:
            """查询指定城市的天气信息"""
            try:
                result = self.weather_tool.get_weather(city_name)
                if result.get("success"):
                    return result.get("data", "获取天气信息失败")
                else:
                    return f"获取{city_name}天气信息失败: {result.get('error', '未知错误')}"
            except Exception as e:
                app_logger.error(f"天气查询工具调用失败: {str(e)}")
                return f"天气查询失败: {str(e)}"
        
        @tool("news_search", args_schema=NewsSearch)
        def news_search(query: str, max_results: int = 5) -> str:
            """搜索新闻和信息"""
            try:
                result = self.search_tool.search_news(query, max_results)
                if result.get("success"):
                    return self.search_tool.format_search_results(result)
                else:
                    return f"搜索失败: {result.get('error', '未知错误')}"
            except Exception as e:
                app_logger.error(f"新闻搜索工具调用失败: {str(e)}")
                return f"搜索失败: {str(e)}"
        
        return [weather_query, news_search]
    
    def _initialize_llm(self) -> ChatOpenAI:
        """初始化语言模型"""
        return ChatOpenAI(
            model="gpt-4o",
            api_key=settings.api.openai_api_key,
            base_url=settings.api.openai_base_url,
            temperature=0.3,
            max_tokens=1000
        )
    
    def _create_general_chain(self):
        """创建通用对话链: prompt | llm | output"""
        prompt = ChatPromptTemplate.from_template("""
        你是一个友好的助手。用户说: {query}
        
        请简洁友好地回答用户的问题。如果用户询问天气，建议他们说"查询XX城市天气"。
        如果用户想搜索信息，建议他们说"搜索XX"。
        """)
        
        return prompt | self.llm | StrOutputParser()

    def chat(self, user_input: str) -> Dict[str, Any]:
        """处理用户输入，使用LLM工具调用机制"""
        start_time = time.time()
        
        try:
            # LLM处理用户输入并自动决定是否需要调用工具
            response = self.llm_with_tools.invoke(user_input)
            
            tools_used = []
            final_response = ""
            
            # 检查是否有工具调用
            if response.tool_calls:
                print(f"🔧 检测到工具调用: {len(response.tool_calls)}个")
                
                # 执行工具调用
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    print(f"📞 调用工具: {tool_name}, 参数: {tool_args}")
                    
                    # 根据工具名称执行相应的工具
                    if tool_name == "weather_query":
                        city_name = tool_args.get('city_name', '')
                        tool_result = self._execute_weather_tool(city_name)
                        tools_used.append("amap_weather_tool")
                    elif tool_name == "news_search":
                        query = tool_args.get('query', '')
                        max_results = tool_args.get('max_results', 5)
                        tool_result = self._execute_search_tool(query, max_results)
                        tools_used.append("tavily_search_tool")
                    else:
                        tool_result = f"未知工具: {tool_name}"
                    
                    # 使用LLM格式化工具结果
                    format_prompt = ChatPromptTemplate.from_template("""
                    用户问题: {user_input}
                    工具结果: {tool_result}
                    
                    请根据工具结果，用自然、友好的语言回答用户的问题。
                    """)
                    
                    format_chain = format_prompt | self.llm | StrOutputParser()
                    final_response = format_chain.invoke({
                        "user_input": user_input,
                        "tool_result": tool_result
                    })
            else:
                # 没有工具调用，使用通用对话链
                final_response = self.general_chain.invoke({"query": user_input})
            
            # 记录对话历史
            self.conversation_history.append({
                "user": user_input,
                "assistant": final_response,
                "timestamp": datetime.now().isoformat(),
                "tools_used": tools_used
            })
            
            # 限制历史长度
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "response": final_response,
                "session_id": self.session_id,
                "processing_time_ms": processing_time,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            app_logger.error(f"对话处理失败: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "response": f"抱歉，处理您的请求时出现了错误: {str(e)}",
                "session_id": self.session_id,
                "processing_time_ms": processing_time,
                "tools_used": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _execute_weather_tool(self, city_name: str) -> str:
        """执行天气查询工具"""
        try:
            result = self.weather_tool.get_weather(city_name)
            if result.get("success"):
                return result.get("data", "获取天气信息失败")
            else:
                return f"获取{city_name}天气信息失败: {result.get('error', '未知错误')}"
        except Exception as e:
            app_logger.error(f"天气查询工具调用失败: {str(e)}")
            return f"天气查询失败: {str(e)}"
    
    def _execute_search_tool(self, query: str, max_results: int = 5) -> str:
        """执行新闻搜索工具"""
        try:
            result = self.search_tool.search_news(query, max_results)
            if result.get("success"):
                return self.search_tool.format_search_results(result['data'])
            else:
                return f"搜索失败: {result.get('error', '未知错误')}"
        except Exception as e:
            app_logger.error(f"新闻搜索工具调用失败: {str(e)}")
            return f"搜索失败: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.conversation_history.copy()
    
    def clear_conversation(self) -> None:
        """清空对话历史"""
        self.conversation_history = []
        app_logger.info(f"对话历史已清空: {self.session_id}")
    
    def end_session(self) -> None:
        """结束会话"""
        app_logger.info(f"会话已结束: {self.session_id}")


def create_qa_agent(session_id: Optional[str] = None) -> QAAgent:
    """创建QA代理实例"""
    return QAAgent(session_id=session_id)