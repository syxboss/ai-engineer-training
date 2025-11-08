import json
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver() 
# InMemorySaver是一个检查点，将 Agent 的状态存储在内存中。
# 在生产环境中，您通常会使用数据库或其他持久存储。请查阅检查点文档以获取更多选项。
# #checkpointer被传递给 Agent 。这使得 Agent 能够在其调用之间持久化其状态。如果您使用LangGraph平台进行部署，该平台将为您提供一个生产就绪的检查点。

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model="deepseek:deepseek-chat",
    tools=[get_weather],
    checkpointer=checkpointer # checkpointer被传递给 Agent 。这使得 Agent 能够在其调用之间持久化其状态。
)

# Run the agent
config = {
    "configurable": {
        "thread_id": "1"   # 配置中提供了一个唯一的thread_id。此ID用于标识对话会话。该值由用户控制，可以是任何字符串.
    }
}

sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config #  Agent 将使用相同的thread_id继续对话。这将允许 Agent 推断用户是在特别询问纽约的天气。
)

# Continue the conversation using the same thread_id
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config 
)

# 自定义序列化函数来处理复杂对象
def serialize_response(obj):
    """将响应对象转换为可序列化的字典"""
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if key.startswith('_'):
                continue
            result[key] = serialize_value(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [serialize_value(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize_value(v) for k, v in obj.items()}
    else:
        return str(obj)

def serialize_value(value):
    """递归序列化值"""
    if hasattr(value, '__dict__'):
        return serialize_response(value)
    elif isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    else:
        try:
            # 尝试序列化值
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            # 如果无法序列化，转换为字符串
            return str(value)

print(json.dumps(serialize_response(ny_response), indent=2, ensure_ascii=False))


