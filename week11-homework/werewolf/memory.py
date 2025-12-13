import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class MemoryManager:
    """
    管理游戏的记忆模块，使用 ChromaDB 作为向量数据库实现 RAG。
    """
    def __init__(self, db_path: str = "./chroma_db"):
        """
        初始化记忆管理器。
        
        Args:
            db_path (str): 向量数据库的持久化路径。
        """
        # 使用 OpenAI 的嵌入模型将文本转换为向量
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        # 初始化 Chroma 向量数据库
        self.vector_store = Chroma(
            collection_name="werewolf_game",
            embedding_function=self.embeddings,
            persist_directory=db_path
        )

    def add_memory(self, player_name: str, content: str, round_num: int, type: str = "observation"):
        """
        为特定玩家添加一条记忆。
        
        Args:
            player_name (str): 记忆所属的玩家名称。
            content (str): 记忆的具体内容（文本）。
            round_num (int): 发生的回合数。
            type (str): 记忆类型（如 "observation", "speech" 等）。
        """
        doc = Document(
            page_content=content,
            metadata={
                "player_name": player_name,
                "round": round_num,
                "type": type
            }
        )
        self.vector_store.add_documents([doc])

    def retrieve_memory(self, player_name: str, query: str, k: int = 3) -> List[str]:
        """
        为特定玩家检索相关的记忆。
        
        Args:
            player_name (str): 需要检索记忆的玩家名称。
            query (str): 查询文本（通常是当前的情境描述）。
            k (int): 返回的最相似记忆数量。
            
        Returns:
            List[str]: 检索到的记忆内容列表。
        """
        # 使用 filter 参数确保玩家只能访问属于自己的记忆（或自己观察到的公共事件）
        results = self.vector_store.similarity_search(
            query,
            k=k,
            filter={"player_name": player_name}
        )
        return [doc.page_content for doc in results]
    
    def clear(self):
        """
        清空记忆库，用于新游戏开始前重置状态。
        """
        try:
            self.vector_store.delete_collection()
        except Exception:
            pass
        # 重新初始化集合，因为 delete_collection 会删除整个集合对象
        self.vector_store = Chroma(
            collection_name="werewolf_game",
            embedding_function=self.embeddings,
            persist_directory=self.vector_store._persist_directory
        )
