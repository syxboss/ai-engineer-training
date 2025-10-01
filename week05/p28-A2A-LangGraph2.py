import logging
import os
import sys
from typing import List, AsyncGenerator

import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

# 导入我们之前创建的SearchAgent
from p28-A2A-LangGraph import SearchAgent, ResponseFormat

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MissingAPIKeyError(Exception):
    """Exception for missing API key."""

class SearchAgentExecutor:
    """SearchAgent的执行器，适配A2A接口"""
    
    def __init__(self, agent: SearchAgent):
        self.agent = agent
    
    async def execute_task(self, task_input: str, session_id: str) -> str:
        """执行搜索任务"""
        return self.agent.invoke(task_input, session_id)
    
    async def execute_task_streaming(self, task_input: str, session_id: str) -> AsyncGenerator[dict, None]:
        """流式执行搜索任务"""
        async for chunk in self.agent.stream(task_input, session_id):
            yield chunk

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10001)
def main(host, port):
    """启动搜索Agent服务器"""
    try:
        # 检查必要的API密钥
        if not os.getenv('TAVILY_API_KEY'):
            raise MissingAPIKeyError('TAVILY_API_KEY environment variable not set.')
        
        if os.getenv('model_source', 'google') == 'google':
            if not os.getenv('GOOGLE_API_KEY'):
                raise MissingAPIKeyError('GOOGLE_API_KEY environment variable not set.')
        else:
            if not os.getenv('TOOL_LLM_URL'):
                raise MissingAPIKeyError('TOOL_LLM_URL environment variable not set.')
            if not os.getenv('TOOL_LLM_NAME'):
                raise MissingAPIKeyError('TOOL_LLM_NAME environment variable not set.')

        # 配置Agent能力
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        
        # 定义搜索技能
        skill = AgentSkill(
            id="search_web",
            name="搜索工具",
            description="搜索web上的相关信息",
            tags=["Web搜索", "互联网搜索"],
            examples=["请搜索最新的黑神话悟空的消息"],
        )

        # 定义Agent卡片
        agent_card = AgentCard(
            name="搜索助手",
            description="搜索Web上的相关信息",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=SearchAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=SearchAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # 初始化Agent和执行器
        search_agent = SearchAgent()
        agent_executor = SearchAgentExecutor(search_agent)

        # 配置HTTP客户端和推送通知
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )

        # 创建请求处理器
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )

        # 创建A2A服务器
        server = A2AStarletteApplication(
            agent_card=agent_card, 
            http_handler=request_handler
        )

        logger.info(f"正在启动服务器，地址：{host}:{port}")
        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f"错误：{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"服务器启动过程中发生错误：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()