"""
LangGraph 重试机制实现示例
实现工具节点的自动重试机制，包含内存写入操作的失败重试处理
"""

import operator
import random
import time
import logging
from typing import Annotated, Sequence, Dict, Any
from dataclasses import dataclass

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 自定义异常类
class MemoryWriteError(Exception):
    """内存写入操作失败异常"""
    pass


class NetworkTimeoutError(Exception):
    """网络超时异常"""
    pass


class ResourceUnavailableError(Exception):
    """资源不可用异常"""
    pass


# 重试策略配置
@dataclass
class RetryPolicy:
    """重试策略配置类"""
    max_attempts: int = 3
    retry_on: tuple = (
        MemoryWriteError,
        NetworkTimeoutError,
        ResourceUnavailableError
    )
    initial_interval: float = 1.0
    backoff_factor: float = 2.0


# 状态定义
class AgentState(TypedDict):
    """Agent状态定义"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    memory_data: Dict[str, Any]
    last_error: str
    operation_success: bool


# 模拟内存存储
class MemoryStorage:
    """模拟内存存储类"""
    
    def __init__(self):
        self.data = {}
        self.failure_rate = 0.6
        
    def write(self, key: str, value: Any) -> bool:
        """
        模拟内存写入操作，可能随机失败
        
        Args:
            key: 存储键
            value: 存储值
            
        Returns:
            bool: 写入是否成功
            
        Raises:
            MemoryWriteError: 内存写入失败
            NetworkTimeoutError: 网络超时
            ResourceUnavailableError: 资源不可用
        """
        if random.random() < self.failure_rate:
            error_type = random.choice([
                MemoryWriteError("内存写入失败：磁盘空间不足"),
                NetworkTimeoutError("网络超时：连接服务器失败"),
                ResourceUnavailableError("资源不可用：内存池已满")
            ])
            logger.error(f"内存写入失败: {error_type}")
            raise error_type
            
        self.data[key] = value
        logger.info(f"成功写入内存: {key} = {value}")
        return True
        
    def read(self, key: str) -> Any:
        """读取内存数据"""
        return self.data.get(key)
        
    def clear(self):
        """清空内存数据"""
        self.data.clear()


# 全局内存存储实例
memory_storage = MemoryStorage()


def write_to_memory(key: str, value: str) -> str:
    """写入内存操作"""
    try:
        success = memory_storage.write(key, value)
        if success:
            return f"成功写入内存: {key}={value}"
        else:
            raise MemoryWriteError(f"写入失败: {key}={value}")
    except Exception as e:
        logger.error(f"内存写入失败: {e}")
        raise


def read_from_memory(key: str) -> str:
    """从内存读取操作"""
    try:
        value = memory_storage.read(key)
        if value is not None:
            return f"读取成功: {key}={value}"
        else:
            return f"键不存在: {key}"
    except Exception as e:
        logger.error(f"内存读取失败: {e}")
        return f"读取失败: {e}"


# 节点函数定义
def memory_write_node(state: AgentState) -> AgentState:
    """
    内存写入节点，包含重试机制
    
    Args:
        state: 当前状态
        
    Returns:
        AgentState: 更新后的状态
    """
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return {
            "messages": [AIMessage(content="无效的消息类型")],
            "operation_success": False,
            "last_error": "无效的消息类型"
        }
    
    content = last_message.content
    if not content.startswith("write ") or "=" not in content:
        return {
            "messages": [AIMessage(content="无效的写入命令格式，请使用: write key=value")],
            "operation_success": False,
            "last_error": "无效的写入命令格式"
        }
    
    try:
        parts = content[6:].split("=", 1)
        key, value = parts[0].strip(), parts[1].strip()
        result = write_to_memory(key, value)
        
        return {
            "messages": [AIMessage(content=f"操作成功: {result}")],
            "operation_success": True,
            "last_error": ""
        }
        
    except Exception as e:
        error_msg = f"操作失败: {str(e)}"
        logger.error(error_msg)
        
        return {
            "messages": [AIMessage(content=error_msg)],
            "operation_success": False,
            "last_error": str(e)
        }


def error_handling_node(state: AgentState) -> AgentState:
    """
    错误处理节点
    
    Args:
        state: 当前状态
        
    Returns:
        AgentState: 更新后的状态
    """
    logger.info("进入错误处理节点")
    
    error_response = f"""操作执行失败

错误信息: {state['last_error']}

建议解决方案:
1. 检查输入格式是否正确
2. 稍后重试操作
3. 联系系统管理员
"""
    
    return {"messages": [AIMessage(content=error_response)]}


def should_handle_error(state: AgentState) -> str:
    """
    条件函数：判断是否需要进行错误处理
    
    Args:
        state: 当前状态
        
    Returns:
        str: 下一个节点的名称
    """
    if state["operation_success"]:
        logger.info("操作成功，流程结束")
        return END
    else:
        logger.info("操作失败，转入错误处理")
        return "error_handling"


# 创建 LangGraph 工作流
def create_retry_workflow() -> StateGraph:
    """
    创建包含重试机制的 LangGraph 工作流
    
    Returns:
        StateGraph: 配置好的状态图
    """
    builder = StateGraph(AgentState)
    
    # 添加节点，使用 retry 参数配置重试策略
    builder.add_node(
        "memory_write",
        memory_write_node,
        retry=RetryPolicy(
            max_attempts=3,
            retry_on=(MemoryWriteError, NetworkTimeoutError, ResourceUnavailableError)
        )
    )
    builder.add_node("error_handling", error_handling_node)
    
    # 设置边
    builder.add_edge(START, "memory_write")
    builder.add_conditional_edges(
        "memory_write",
        should_handle_error,
        {
            END: END,
            "error_handling": "error_handling"
        }
    )
    builder.add_edge("error_handling", END)
    
    return builder


# 示例演示类
class RetryDemo:
    """重试机制演示类"""
    
    def __init__(self):
        builder = create_retry_workflow()
        self.app = builder.compile(checkpointer=MemorySaver())
        
    def run_demo_scenario(self, scenario_name: str, command: str, failure_rate: float = 0.6):
        """
        运行演示场景
        
        Args:
            scenario_name: 场景名称
            command: 执行命令
            failure_rate: 失败率
        """
        print(f"\n{'='*60}")
        print(f"演示场景: {scenario_name}")
        print(f"执行命令: {command}")
        print(f"失败率设置: {failure_rate * 100}%")
        print(f"{'='*60}")
        
        memory_storage.failure_rate = failure_rate
        memory_storage.clear()
        
        initial_state = {
            "messages": [HumanMessage(content=command)],
            "memory_data": {},
            "last_error": "",
            "operation_success": False
        }
        
        config = {"configurable": {"thread_id": f"demo_{int(time.time())}"}}
        
        try:
            result = self.app.invoke(initial_state, config)
            
            print("\n执行结果:")
            print(f"操作成功: {result['operation_success']}")
            
            print("\n最终消息:")
            final_message = result['messages'][-1]
            print(f"  {final_message.content}")
            
            if memory_storage.data:
                print("\n内存数据:")
                for key, value in memory_storage.data.items():
                    print(f"  {key} = {value}")
            
        except Exception as e:
            print(f"\n演示执行异常: {e}")
        
        print(f"\n{'='*60}\n")


def main():
    """主函数：运行完整的重试机制演示"""
    print("LangGraph 重试机制实现示例")
    print("=" * 60)
    
    demo = RetryDemo()
    
    # 场景1: 正常执行流程（低失败率）
    demo.run_demo_scenario(
        "正常执行流程",
        "write user_name=张三",
        failure_rate=0.0
    )
    
    # 场景2: 重试成功场景（中等失败率）
    demo.run_demo_scenario(
        "重试成功场景",
        "write session_id=abc123",
        failure_rate=0.4
    )
    
    # 场景3: 重试失败后的处理（高失败率）
    demo.run_demo_scenario(
        "重试失败处理",
        "write config_data=重要配置",
        failure_rate=0.9
    )
    
    # 场景4: 无效命令格式
    demo.run_demo_scenario(
        "无效命令处理",
        "invalid command format",
        failure_rate=0.0
    )
    
    print("所有演示场景执行完成！")
    
    # 显示重试策略配置信息
    print("\n重试策略配置:")
    policy = RetryPolicy()
    print(f"  最大重试次数: {policy.max_attempts}")
    print(f"  初始间隔时间: {policy.initial_interval}秒")
    print(f"  退避因子: {policy.backoff_factor}")
    print(f"  可重试异常: {[exc.__name__ for exc in policy.retry_on]}")


if __name__ == "__main__":
    main()