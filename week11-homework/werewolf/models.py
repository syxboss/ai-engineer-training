from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Player(BaseModel):
    """
    定义玩家数据模型
    """
    name: str  # 玩家名称
    role: Literal["villager", "werewolf"]  # 玩家角色：村民或狼人
    status: Literal["alive", "dead"] = "alive"  # 玩家状态：存活或死亡
    personality: str = ""  # 玩家性格描述，用于影响 Agent 的发言风格
    
    # 标记是否为 AI 玩家（当前版本默认为 True）
    is_ai: bool = True 

class GameState(BaseModel):
    """
    定义游戏全局状态模型
    """
    round: int = 1  # 当前回合数
    phase: Literal["night", "day_discussion", "day_voting", "game_over"] = "night"  # 当前游戏阶段
    players: List[Player]  # 所有参与游戏的玩家列表
    history: List[str] = Field(default_factory=list) # 游戏历史记录（文本格式），用于提供给 Agent 上下文
    winners: Optional[Literal["villagers", "werewolves"]] = None  # 获胜方，None 表示游戏仍在进行
    
    # 记录当前回合死亡的玩家（用于天亮时公布）
    recent_deaths: List[str] = Field(default_factory=list)
    
    # 记录当前回合的投票情况：投票人 -> 被投票人
    current_votes: dict[str, str] = Field(default_factory=dict) 
