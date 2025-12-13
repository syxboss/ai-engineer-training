import os
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from werewolf.models import Player, GameState
from werewolf.memory import MemoryManager

class VoteDecision(BaseModel):
    target: str = Field(description="The name of the player you vote for. Must be one of the valid options.")
    reasoning: str = Field(description="The reasoning behind your vote.")

class NightActionDecision(BaseModel):
    target: str = Field(description="The name of the player you choose to kill. Must be one of the valid options or 'Sleep'.")
    reasoning: str = Field(description="The reasoning behind your choice.")

class Reflection(BaseModel):
    content: str = Field(description="The content of your inner monologue/reflection. Must be enclosed in <think> tags.")

class WerewolfGameAgent:
    """
    狼人杀游戏智能体，代表一个玩家。
    包含角色扮演、发言生成、投票决策和夜晚行动逻辑。
    """
    def __init__(self, player: Player, memory_manager: MemoryManager):
        """
        初始化智能体。
        
        Args:
            player (Player): 玩家信息对象。
            memory_manager (MemoryManager): 记忆管理器实例。
        """
        self.player = player
        self.memory = memory_manager
        # 使用 GPT-4o-mini 作为推理引擎
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.7,
            api_key=os.environ.get("OPENAI_API_KEY")
        )

    def _get_role_prompt(self) -> str:
        """
        根据玩家角色生成系统提示词 (System Prompt)。
        包含角色目标、性格特征和行为准则。
        """
        base_prompt = (
            f"你是 {self.player.name}，一个真实的人类玩家，正在和朋友们面对面玩狼人杀游戏。\n"
            f"你的性格特征：{self.player.personality}\n"
            "【重要行为准则】\n"
            "1. 必须像真人一样说话。使用自然的口语、语气词（如'嗯...'、'我觉得'、'那个'），表达情感（紧张、怀疑、愤怒、放松）。\n"
            "2. 严禁出现任何'出戏'的词汇。绝对不要说你是'AI'、'语言模型'、'程序'或提到'模拟'、'算法'。\n"
            "3. 沉浸在游戏场景中。你就在一个房间里和其他人坐在一起。\n"
            "4. 不要使用过于书面化或机械的语言。不要列出带编号的清单，除非你在非常正式地梳理逻辑。\n"
        )

        if self.player.role == "villager":
            return base_prompt + (
                "你的身份是【村民】。\n"
                "你的目标是找出并投票处决所有狼人。\n"
                "你不知道谁是狼人，你必须依靠逻辑和直觉。\n"
                "请仔细分析其他人的发言，寻找逻辑漏洞或矛盾之处。\n"
                "发言要符合逻辑，同时体现你的性格特征。\n"
                "重要：不要讨论物理上的不在场证明（如'我昨晚在家睡觉'），因为在游戏中所有人在夜晚都闭眼或行动，这种说法没有意义。\n"
                "重点关注投票行为、发言逻辑和情绪反应。"
            )
        elif self.player.role == "werewolf":
            return base_prompt + (
                "你的身份是【狼人】。\n"
                "你的目标是消灭所有村民，同时不要被发现。\n"
                "你需要伪装成村民，混淆视听。\n"
                "在白天发言时，要转移怀疑对象，必要时可以栽赃陷害。\n"
                "尽量让自己的发言听起来像个无辜的村民。\n"
                "与其他狼人配合（虽然你可能无法直接沟通，但要尝试理解局势）。\n"
                "重要：不要编造物理上的不在场证明（如'我昨晚在家'），这在狼人杀中很假。\n"
                "你应该质疑他人的逻辑，或者引导大家去怀疑某个好人。"
            )
        return ""

    def speak(self, game_state: GameState, context: str = "") -> str:
        """
        在讨论阶段生成发言。
        
        Args:
            game_state (GameState): 当前游戏状态。
            context (str): 当前的对话上下文（例如上一位玩家的发言）。
            
        Returns:
            str: 玩家的发言内容。
        """
        # RAG: 检索相关记忆
        query = f"第 {game_state.round} 回合的当前局势: {context}"
        memories = self.memory.retrieve_memory(self.player.name, query)
        memory_context = "\n".join(memories) if memories else "暂无具体相关记忆。"

        system_prompt = self._get_role_prompt()
        
        alive_players = [p.name for p in game_state.players if p.status == "alive"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("user", 
             "当前回合: {round}\n"
             "当前阶段: {phase}\n"
             "存活玩家: {alive_players}\n"
             "近期历史记录:\n{history}\n\n"
             "你的记忆/知识:\n{memory_context}\n\n"
             "当前对话上下文: {context}\n"
             "请发表你的观点（保持简练，50字以内）。"
             "请不要在回复中包含你的名字前缀。"
             "如果你是狼人，请务必隐藏好身份。"
            )
        ])

        chain = prompt | self.llm | StrOutputParser()
        
        # 获取最近的历史记录作为上下文窗口
        recent_history = "\n".join(game_state.history[-10:])
        
        response = chain.invoke({
            "system_prompt": system_prompt,
            "round": game_state.round,
            "phase": game_state.phase,
            "alive_players": ", ".join(alive_players),
            "history": recent_history,
            "memory_context": memory_context,
            "context": context
        })
        
        return response.strip()

    def reflect(self, game_state: GameState) -> str:
        """
        进行反思和推理 (Chain of Thought)。
        
        Args:
            game_state (GameState): 当前游戏状态。
            
        Returns:
            str: 反思内容，包含在 <think> 标签中。
        """
        query = f"第 {game_state.round} 回合局势分析"
        memories = self.memory.retrieve_memory(self.player.name, query)
        memory_context = "\n".join(memories) if memories else "暂无具体相关记忆。"
        
        system_prompt = self._get_role_prompt()
        alive_players = [p.name for p in game_state.players if p.status == "alive"]
        recent_history = "\n".join(game_state.history[-15:])
        
        user_prompt = ""
        if self.player.role == "villager":
            user_prompt = (
                "现在是反思阶段（内心独白）。\n"
                "请根据最近的讨论和历史记录，仔细分析局势。\n"
                "1. 谁的发言最可疑？是否存在逻辑漏洞或矛盾？\n"
                "2. 谁可能是狼人？为什么？\n"
                "3. 你接下来的计划是什么？\n"
                "请用 <think>...</think> 标签包裹你的内心独白。这是你的秘密想法。"
            )
        else: # werewolf
            user_prompt = (
                "现在是反思阶段（内心独白）。\n"
                "作为狼人，你需要评估当前的伪装效果。\n"
                "1. 有没有人开始怀疑你了？\n"
                "2. 谁是目前对你最大的威胁（例如逻辑清晰的村民）？\n"
                "3. 你接下来应该栽赃谁，或者如何转移注意力？\n"
                "请用 <think>...</think> 标签包裹你的内心独白。这是你的秘密想法。"
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("user", 
             "当前回合: {round}\n"
             "存活玩家: {alive_players}\n"
             "近期历史记录:\n{history}\n\n"
             "你的记忆:\n{memory_context}\n\n" + 
             user_prompt
            )
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "system_prompt": system_prompt,
            "round": game_state.round,
            "alive_players": ", ".join(alive_players),
            "history": recent_history,
            "memory_context": memory_context
        })
        
        # 简单的后处理，确保有 <think> 标签
        content = response.strip()
        if not content.startswith("<think>"):
            content = f"<think>{content}</think>"
        
        return content

    def vote(self, game_state: GameState) -> str:
        """
        进行投票决策。
        
        Args:
            game_state (GameState): 当前游戏状态。
            
        Returns:
            str: 投票目标的玩家名称。
        """
        alive_players = [p.name for p in game_state.players if p.status == "alive"]
        # 排除自己
        options = [p for p in alive_players if p != self.player.name]
        
        if not options:
            return "弃票"

        system_prompt = self._get_role_prompt()
        recent_history = "\n".join(game_state.history[-15:])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("user", 
             "现在是投票时间。 \n"
             "存活玩家: {alive_players}\n"
             "近期历史记录:\n{history}\n"
             "根据之前的讨论，你想要投票处决谁？"
             "可选项: {options}。"
             "如果你不确定，请投给最可疑的人。"
            )
        ])
        
        # 使用结构化输出以保证稳定性
        structured_llm = self.llm.with_structured_output(VoteDecision)
        chain = prompt | structured_llm
        
        try:
            decision = chain.invoke({
                "system_prompt": system_prompt,
                "alive_players": ", ".join(alive_players),
                "history": recent_history,
                "options": ", ".join(options)
            })
            
            # 验证输出是否有效
            vote_target = decision.target
            # 模糊匹配以防止轻微拼写错误
            for opt in options:
                if opt in vote_target:
                    return opt
            # 如果完全不匹配，回退到第一个选项（或者最可疑的逻辑，这里简化处理）
            return options[0]
            
        except Exception as e:
            print(f"Voting Error for {self.player.name}: {e}")
            return options[0]

    def night_action(self, game_state: GameState) -> str:
        """
        执行夜晚行动。
        狼人：选择击杀目标。
        村民：无行动（或睡觉）。
        
        Args:
            game_state (GameState): 当前游戏状态。
            
        Returns:
            str: 目标玩家名称（如果是狼人），否则为 "Sleep"。
        """
        if self.player.role != "werewolf":
            return "Sleep"
            
        alive_villagers = [p.name for p in game_state.players if p.status == "alive" and p.role != "werewolf"]
        if not alive_villagers:
             return "None"

        system_prompt = self._get_role_prompt()
        recent_history = "\n".join(game_state.history[-10:])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("user", 
             "现在是夜晚。你需要选择一名村民进行击杀。\n"
             "近期历史: {history}\n"
             "存活村民 (可选项): {options}\n"
             "请选择你的目标。"
            )
        ])
        
        structured_llm = self.llm.with_structured_output(NightActionDecision)
        chain = prompt | structured_llm
        
        try:
            decision = chain.invoke({
                "system_prompt": system_prompt,
                "history": recent_history,
                "options": ", ".join(alive_villagers)
            })
            
            target = decision.target
            for opt in alive_villagers:
                if opt in target:
                    return opt
            return alive_villagers[0]
            
        except Exception as e:
            print(f"Night Action Error for {self.player.name}: {e}")
            return alive_villagers[0]
