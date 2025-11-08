import os
from llama_index.core.settings import Settings
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding
from dotenv import load_dotenv


# 加载 .env 文件
load_dotenv()

# --- API Keys and Credentials ---
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([DASHSCOPE_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError("请检查 .env 文件，确保所有环境变量都已正确设置。")

# --- LlamaIndex Global Settings ---
# 使用通义千问 qwen-plus 作为 LLM
Settings.llm = DashScope(model_name="qwen-plus", api_key=DASHSCOPE_API_KEY)
# 使用通义千问的文本嵌入模型
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2", api_key=DASHSCOPE_API_KEY
)

# --- Data Paths ---
DATA_DIR = "data"
COMPANY_DOC_PATH = os.path.join(DATA_DIR, "companies.txt")
SHAREHOLDER_CSV_PATH = os.path.join(DATA_DIR, "shareholders.csv")

# --- Vector Index Settings ---
INDEX_DIR = "vector_index"

# --- Neo4j Graph Settings ---
NEO4J_DATABASE = "neo4j"