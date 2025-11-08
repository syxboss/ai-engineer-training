from langchain_openai import ChatOpenAI
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from typing import Any

model = ChatOpenAI(model="gpt-4o")

# 定义一个简单的工具函数
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 定义工具列表
tools = [get_weather]

summarization_node = SummarizationNode( 
    token_counter=count_tokens_approximately,
    model=model,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages",
)

class State(AgentState):
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: dict[str, Any]  


checkpointer = InMemorySaver() 

agent = create_react_agent(
    model=model,
    tools=tools,
    pre_model_hook=summarization_node, 
    state_schema=State, 
    checkpointer=checkpointer,
)