from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool

# 初始化模型
model = ChatTongyi(
    model="qwen-max",
    temperature=0.7,
    streaming=True
)

# 定义一个简单的工具函数
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 定义工具列表
tools = [get_weather]

# This function will be called every time before the node that calls LLM
# 实现窗口记忆机制：保持最近的n条消息，超出时自动删除旧消息
def pre_model_hook(state):
    """
    窗口记忆机制的预处理钩子函数
    
    功能：
    1. 维护固定大小的消息窗口
    2. 当消息数量超过限制时，自动移除较旧的消息
    3. 保持窗口内消息的时间顺序
    
    参数：
    - strategy="last": 保留最后的消息
    - max_tokens=384: 最大token数限制
    - start_on="human": 从人类消息开始
    - end_on=("human", "tool"): 在人类或工具消息结束
    """
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",  # 保留最后的消息（最新的消息）
        token_counter=count_tokens_approximately,
        max_tokens=384,  # 窗口大小限制（以token计算）
        start_on="human",  # 确保从人类消息开始
        end_on=("human", "tool"),  # 在人类或工具消息结束
    )
    return {"llm_input_messages": trimmed_messages}

# 初始化检查点保存器（内存中保存状态）
checkpointer = InMemorySaver()

# 创建具有窗口记忆机制的React Agent
agent = create_react_agent(
    model,
    tools,
    pre_model_hook=pre_model_hook,  # 使用窗口记忆预处理钩子
    checkpointer=checkpointer,
)

# 示例使用
if __name__ == "__main__":
    print("窗口记忆机制演示")
    print("=" * 50)
    print("特性:")
    print("- 保持最近384个token的对话历史")
    print("- 自动删除超出窗口的旧消息")
    print("- 保持消息的时间顺序")
    print("- 确保对话的连续性")
    print()
    
    # 配置
    config = {
        "configurable": {
            "thread_id": "window-demo-1"
        }
    }
    
    # 演示对话
    demo_messages = [
        "你好，我是张三，今年25岁",
        "我住在北京，是一名软件工程师",
        "我喜欢编程和阅读",
        "请问今天北京的天气怎么样？",
        "你还记得我的名字和职业吗？"
    ]
    
    print("开始演示对话:")
    print("-" * 30)
    
    for i, message in enumerate(demo_messages, 1):
        print(f"\n[轮次 {i}] 用户: {message}")
        
        try:
            # 调用agent处理消息
            response = agent.invoke(
                {"messages": [("user", message)]},
                config=config
            )
            
            # 显示AI回复
            if response and "messages" in response:
                ai_message = response["messages"][-1]
                if hasattr(ai_message, 'content'):
                    print(f"[轮次 {i}] AI: {ai_message.content}")
                else:
                    print(f"[轮次 {i}] AI: {ai_message}")
            
        except Exception as e:
            print(f"[轮次 {i}] 错误: {e}")
    
    print("\n" + "=" * 50)
    print("窗口记忆机制演示完成！")
    print("注意：由于窗口限制，早期的对话可能会被自动删除")