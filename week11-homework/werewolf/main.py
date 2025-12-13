import os
import json
import time
import argparse
import random
from langchain_community.callbacks.manager import get_openai_callback
from werewolf.models import Player, GameState
from werewolf.memory import MemoryManager
from werewolf.game_graph import GameGraphBuilder
from rich.console import Console

# 初始化 Rich 控制台，用于美化输出
console = Console()

def main():
    """
    游戏主入口函数。
    负责初始化游戏、运行模拟、统计成本并输出日志。
    """
    parser = argparse.ArgumentParser(description="狼人杀 AI 模拟")
    parser.add_argument("--wolves", type=int, default=1, help="狼人数量 (默认为 1)")
    parser.add_argument("--players", type=int, default=6, help="玩家总数 (默认为 6，最大 10)")
    args = parser.parse_args()

    num_wolves = args.wolves
    num_players = args.players
    
    # 1. 初始化记忆管理器
    # 强制清除 ChromaDB 目录以确保干净的开始
    import shutil
    if os.path.exists("./chroma_db"):
        try:
            shutil.rmtree("./chroma_db")
        except Exception as e:
            console.print(f"[yellow]警告: 无法完全删除旧数据库: {e}[/yellow]")

    # 使用 ChromaDB 存储游戏过程中的对话和事件
    memory_manager = MemoryManager(db_path="./chroma_db")
    # memory_manager.clear() # 已通过物理删除清除
    
    # 2. 初始化玩家列表
    # 预定义的玩家名称池 (扩展至 10 人)
    all_names = [
        "Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", 
        "Grace", "Heidi", "Ivan", "Judy"
    ]
    
    # 预定义的性格池 (扩展至 10+ 种)
    personalities = [
        "谨慎且讲逻辑。相信事实。",
        "激进且大声。喜欢快速指控他人。",
        "观察力敏锐且安静。只有确信时才发言。",
        "紧张且健谈。喜欢问很多问题。",
        "具有欺骗性且善于操纵。假装是领导者。",
        "追随者。为了融入群体而附和大多数人。",
        "情绪化且容易被煽动。",
        "冷静且客观，喜欢分析概率。",
        "幽默且喜欢缓和气氛，但有时会显得不严肃。",
        "多疑且悲观，总觉得好人要输了。",
        "自信且傲慢，认为自己总是对的。"
    ]
    
    if num_players > len(all_names):
        console.print(f"[bold red]错误: 玩家数量不能超过 {len(all_names)} 人！[/bold red]")
        return

    if num_wolves >= num_players:
        console.print("[bold red]错误: 狼人数量不能超过玩家总数！[/bold red]")
        return

    # 选取指定数量的玩家
    names = all_names[:num_players]

    # 分配角色和性格
    players = []
    
    # 随机打乱名字和性格，确保每次都不一样
    random.shuffle(names)
    random.shuffle(personalities)
    
    for i, name in enumerate(names):
        # 简单的分配逻辑：最后的 num_wolves 个玩家是狼人
        role = "villager"
        if i >= len(names) - num_wolves:
            role = "werewolf"
        
        # 分配一个性格
        personality = personalities[i % len(personalities)]
        
        players.append(Player(name=name, role=role, personality=personality))
    
    # 打印角色分配情况 (调试用，实际游戏中通常保密，但模拟器需要知道)
    console.print("[bold yellow]角色分配:[/bold yellow]")
    for p in players:
        console.print(f"{p.name}: {p.role} ({p.personality})")

    # 创建初始游戏状态
    initial_state = GameState(players=players)
    
    # 3. 构建游戏流程图
    # 使用 LangGraph 定义状态机
    builder = GameGraphBuilder(memory_manager)
    game = builder.build()
    
    console.print("\n[bold green]开始狼人杀游戏模拟...[/bold green]")
    
    start_time = time.time()
    
    # 4. 运行游戏并追踪成本
    # get_openai_callback 用于统计 LangChain 调用 OpenAI API 的 Token 消耗和费用
    with get_openai_callback() as cb:
        # 启动图执行
        # recursion_limit 控制最大递归深度，防止游戏死循环
        # 狼人杀通常会有多个昼夜交替，因此需要较高的限制 (默认 25，这里设为 50)
        final_state = game.invoke(initial_state, config={"recursion_limit": 50})
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 5. 输出结果
        console.print(f"\n[bold red]游戏结束！[/bold red]")
        
        winner_text = "未知"
        if final_state['winners'] == "villagers":
            winner_text = "村民阵营 (好人)"
        elif final_state['winners'] == "werewolves":
            winner_text = "狼人阵营"
            
        console.print(f"获胜方: {winner_text}")
        
        # 6. 生成游戏日志
        # 包含完整对局历史、获胜方以及性能指标
        log_content = {
            "history": final_state['history'],
            "winners": final_state['winners'],
            "metrics": {
                "duration_seconds": round(duration, 2),
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost_usd": cb.total_cost
            }
        }
        
        with open("game_log.json", "w") as f:
            json.dump(log_content, f, indent=2, ensure_ascii=False)
            
        console.print(f"游戏日志已保存至 game_log.json")
        console.print(f"\n[bold blue]复杂度和成本分析:[/bold blue]")
        console.print(f"Token 总数: {cb.total_tokens}")
        console.print(f"总成本: ${cb.total_cost:.4f}")
        console.print(f"耗时: {duration:.2f}s")

if __name__ == "__main__":
    main()
