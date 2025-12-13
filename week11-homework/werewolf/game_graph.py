from typing import Annotated, Literal
from langgraph.graph import StateGraph, END
from werewolf.models import GameState, Player
from werewolf.agents import WerewolfGameAgent
from werewolf.memory import MemoryManager
import random

# 获取 Agent 实例的辅助函数
def get_agent(player: Player, memory: MemoryManager) -> WerewolfGameAgent:
    return WerewolfGameAgent(player, memory)

class GameGraphBuilder:
    """
    构建狼人杀游戏流程图 (StateGraph)。
    定义了游戏的各个节点（阶段）和转换逻辑。
    """
    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
        self.graph = StateGraph(GameState)

    def night_node(self, state: GameState) -> GameState:
        """
        【夜晚阶段】节点
        狼人行动：所有存活狼人投票选择一名玩家进行击杀。
        """
        print("\n--- 夜晚阶段 ---")
        state.phase = "night"
        
        alive_werewolves = [p for p in state.players if p.role == "werewolf" and p.status == "alive"]
        target = None
        
        if alive_werewolves:
            votes = {}
            print(f"狼人阵营 ({len(alive_werewolves)} 人) 正在行动...")
            
            # 简单的多狼人决策：投票制
            for wolf in alive_werewolves:
                agent = get_agent(wolf, self.memory)
                vote = agent.night_action(state)
                if vote not in ["Sleep", "None"]:
                    votes[vote] = votes.get(vote, 0) + 1
                    print(f"  狼人 {wolf.name} 提议杀害: {vote}")
            
            if votes:
                # 选择票数最高的目标
                target = max(votes, key=votes.get)
                print(f"狼人一致/多数决定选择了目标: {target}")
            else:
                print("狼人未达成有效击杀目标。")
        
        # 处理死亡逻辑
        killed = []
        if target:
            for p in state.players:
                if p.name == target and p.status == "alive":
                    p.status = "dead"
                    killed.append(p.name)
                    state.history.append(f"第 {state.round} 回合夜晚: {p.name} 被杀害。")
                    # 添加死亡记忆
                    self.memory.add_memory(p.name, f"我在第 {state.round} 回合被杀害。", state.round)
        
        state.recent_deaths = killed
        return state

    def day_announce_node(self, state: GameState) -> GameState:
        """
        【天亮公布】节点
        主持人公布昨晚的死亡名单。
        """
        print(f"\n--- 第 {state.round} 回合天亮 ---")
        state.phase = "day_discussion"
        
        if state.recent_deaths:
            announcement = f"昨晚，{', '.join(state.recent_deaths)} 死亡。"
        else:
            announcement = "昨晚是平安夜，无人死亡。"
            
        print(f"主持人: {announcement}")
        state.history.append(f"第 {state.round} 回合: 主持人宣布: {announcement}")
        
        # 公布死亡后立即检查是否有获胜方
        winner = self._check_win(state)
        if winner:
            state.winners = winner
            
        return state

    def discussion_node(self, state: GameState) -> GameState:
        """
        【讨论阶段】节点
        存活玩家依次发言。
        """
        if state.winners:
            return state
            
        print("\n--- 讨论阶段 ---")
        alive_players = [p for p in state.players if p.status == "alive"]
        
        # 打乱发言顺序，增加随机性和公平性
        random.shuffle(alive_players) 
        
        discussion_log = []
        for player in alive_players:
            agent = get_agent(player, self.memory)
            # 上下文取最近的几条发言，避免 Prompt 过长
            context = "\n".join(discussion_log[-3:]) if discussion_log else "讨论开始。"
            
            speech = agent.speak(state, context)
            print(f"{player.name}: {speech}")
            
            log_entry = f"{player.name}: {speech}"
            discussion_log.append(log_entry)
            state.history.append(f"第 {state.round} 回合讨论: {log_entry}")
            
            # 将发言记录到所有在场玩家的记忆中
            for p in state.players:
                if p.status == "alive":
                    self.memory.add_memory(p.name, f"{player.name} 说了: {speech}", state.round)
        
        return state

    def reflection_node(self, state: GameState) -> GameState:
        """
        【反思阶段】节点
        在讨论结束后、投票前，让所有存活玩家进行一轮内心独白和推理。
        """
        if state.winners:
            return state
            
        print("\n--- 反思阶段 (内心独白) ---")
        alive_players = [p for p in state.players if p.status == "alive"]
        
        for player in alive_players:
            agent = get_agent(player, self.memory)
            reflection = agent.reflect(state)
            
            print(f"{player.name} 反思: {reflection}")
            
            # 将反思存入记忆，作为“内心的秘密”
            self.memory.add_memory(player.name, f"反思/推理: {reflection}", state.round, type="reflection")
            
        return state

    def voting_node(self, state: GameState) -> GameState:
        """
        【投票阶段】节点
        存活玩家进行投票，票数最高者被处决。
        """
        if state.winners:
            return state

        print("\n--- 投票阶段 ---")
        state.phase = "day_voting"
        
        alive_players = [p for p in state.players if p.status == "alive"]
        votes = {}
        
        # 收集所有玩家的投票
        for player in alive_players:
            agent = get_agent(player, self.memory)
            vote = agent.vote(state)
            votes[player.name] = vote
            print(f"{player.name} 投票给 {vote}")
            state.history.append(f"第 {state.round} 回合投票: {player.name} 投给了 {vote}")
            self.memory.add_memory(player.name, f"我投给了 {vote}", state.round)
        
        state.current_votes = votes
        
        # 统计票数
        vote_counts = {}
        for target in votes.values():
            vote_counts[target] = vote_counts.get(target, 0) + 1
            
        # 寻找票数最高者
        if not vote_counts:
            return state
            
        max_votes = max(vote_counts.values())
        candidates = [k for k, v in vote_counts.items() if v == max_votes]
        
        eliminated = None
        if len(candidates) == 1:
            eliminated = candidates[0]
            if eliminated != "弃票":
                print(f"玩家 {eliminated} 以 {max_votes} 票被处决。")
                state.history.append(f"第 {state.round} 回合结果: {eliminated} 被处决。")
            else:
                print("多数人选择弃票，无人出局。")
        else:
            # 平票情况：无人出局（简化规则）
            print(f"平票 ({', '.join(candidates)})，无人出局。")
            state.history.append(f"第 {state.round} 回合结果: 平票，无人被处决。")
            
        if eliminated and eliminated != "弃票":
            for p in state.players:
                if p.name == eliminated:
                    p.status = "dead"
                    
        # 投票处决后再次检查胜利条件
        winner = self._check_win(state)
        if winner:
            state.winners = winner
            
        return state

    def round_end_node(self, state: GameState) -> GameState:
        """
        【回合结束】节点
        递增回合数，准备进入下一轮夜晚。
        """
        if state.winners:
            return state
            
        state.round += 1
        return state

    def _check_win(self, state: GameState) -> Literal["villagers", "werewolves", None]:
        """
        检查胜利条件。
        
        Returns:
            "villagers": 狼人全灭，村民胜利。
            "werewolves": 狼人数量 >= 村民数量，狼人胜利。
            None: 游戏继续。
        """
        alive_werewolves = [p for p in state.players if p.role == "werewolf" and p.status == "alive"]
        alive_villagers = [p for p in state.players if p.role == "villager" and p.status == "alive"]
        
        if not alive_werewolves:
            return "villagers"
        if len(alive_werewolves) >= len(alive_villagers):
            return "werewolves"
        return None

    def build(self):
        """
        构建并编译状态图。
        """
        self.graph.add_node("night", self.night_node)
        self.graph.add_node("announce", self.day_announce_node)
        self.graph.add_node("discussion", self.discussion_node)
        self.graph.add_node("reflection", self.reflection_node)
        self.graph.add_node("voting", self.voting_node)
        self.graph.add_node("round_end", self.round_end_node)
        
        # 设置入口点
        self.graph.set_entry_point("night")
        
        # 添加边（流程控制）
        self.graph.add_edge("night", "announce")
        
        # 条件边：如果游戏结束，直接终止；否则进入讨论
        def check_game_over(state: GameState):
            if state.winners:
                return END
            return "discussion"
            
        self.graph.add_conditional_edges("announce", check_game_over)
        
        self.graph.add_edge("discussion", "reflection")
        self.graph.add_edge("reflection", "voting")
        
        # 条件边：如果游戏结束，直接终止；否则进入回合结算
        def check_game_over_vote(state: GameState):
            if state.winners:
                return END
            return "round_end"
            
        self.graph.add_conditional_edges("voting", check_game_over_vote)
        
        self.graph.add_edge("round_end", "night")
        
        return self.graph.compile()
