import logging
import numpy as np
import os
import dashscope
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# 设置通义千问API密钥
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")

class QwenEmbedding:
    """通义千问文本嵌入服务"""
    
    def __init__(self, model_name="text-embedding-v3"):
        self.model_name = model_name
        
    def encode(self, texts):
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            from dashscope import TextEmbedding
            
            response = TextEmbedding.call(
                model=self.model_name,
                input=texts
            )
            
            if response.status_code == 200:
                embeddings = []
                for output in response.output['embeddings']:
                    embeddings.append(np.array(output['embedding']))
                
                return embeddings[0] if len(embeddings) == 1 else embeddings
            else:
                logger.error(f"通义千问embedding调用失败: {response}")
                # 降级到本地模型
                fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
                return fallback_model.encode(texts)
                
        except Exception as e:
            logger.error(f"通义千问embedding异常: {e}")
            # 降级到本地模型
            fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
            return fallback_model.encode(texts)