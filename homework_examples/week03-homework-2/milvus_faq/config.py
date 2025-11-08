import os
from llama_index.core.settings import Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding
from dotenv import load_dotenv


# 加载 .env 文件中的环境变量
load_dotenv()

# 确保你的通义千问 API Key 已经设置在环境变量中
# 环境变量名称：DASHSCOPE_API_KEY
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# --- 模型和嵌入配置 ---
# 使用通义千问的 'qwen-plus' 作为 LLM (虽然本项目主要用嵌入，但设置好以备扩展)
# llm = DashScope(model_name="qwen-plus", api_key=DASHSCOPE_API_KEY)

# 使用通义千问的文本嵌入模型
EMBED_MODEL = DashScopeEmbedding(
    model_name="text-embedding-v2", 
    api_key=DASHSCOPE_API_KEY
)

# --- LlamaIndex 全局设置 ---
# 将嵌入模型配置到 LlamaIndex 的全局设置中
Settings.embed_model = EMBED_MODEL
# Settings.llm = llm # 如果需要LLM进行答案合成，取消此行注释
Settings.chunk_size = 512
Settings.chunk_overlap = 20

# --- 数据和索引路径配置 ---
DATA_DIR = "data"
FAQ_FILE = os.path.join(DATA_DIR, "faqs.csv")
INDEX_DIR = "vector_index" # Milvus Lite 数据将存储在这里

# --- Milvus 配置 ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_URI = "./milvus_demo.db" # Milvus Lite 使用本地文件
COLLECTION_NAME = "faq_collection"
DIMENSION = 1536 # 通义千问 text-embedding-v2 模型的维度
