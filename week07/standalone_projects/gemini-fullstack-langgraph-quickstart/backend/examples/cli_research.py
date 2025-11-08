import argparse
from langchain_core.messages import HumanMessage
from agent.graph import graph


def main() -> None:
    """
    从命令行运行研究代理
    
    这个函数提供了一个命令行接口来直接使用LangGraph研究代理，
    允许用户通过命令行参数配置研究参数并获得研究结果。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="运行LangGraph研究代理")
    
    # 必需参数：研究问题
    parser.add_argument("question", help="研究问题")
    
    # 可选参数：初始搜索查询数量
    parser.add_argument(
        "--initial-queries",
        type=int,
        default=3,
        help="初始搜索查询的数量（默认：3）",
    )
    
    # 可选参数：最大研究循环次数
    parser.add_argument(
        "--max-loops",
        type=int,
        default=2,
        help="最大研究循环次数（默认：2）",
    )
    
    # 可选参数：推理模型选择
    parser.add_argument(
        "--reasoning-model",
        default="gemini-2.5-pro-preview-05-06",
        help="用于最终答案的模型（默认：gemini-2.5-pro-preview-05-06）",
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    # 构建初始状态，包含用户问题和配置参数
    state = {
        "messages": [HumanMessage(content=args.question)],  # 用户的研究问题
        "initial_search_query_count": args.initial_queries,  # 初始查询数量
        "max_research_loops": args.max_loops,  # 最大循环次数
        "reasoning_model": args.reasoning_model,  # 推理模型
    }

    # 调用图执行研究流程
    result = graph.invoke(state)
    
    # 提取并打印最终的研究结果
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)  # 打印最后一条消息（AI的最终答案）


# 如果作为主程序运行，则执行main函数
if __name__ == "__main__":
    main()
