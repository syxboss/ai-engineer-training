# pip install vanna
import vanna
import os

from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

vn = MyVanna(config={'api_key': os.getenv('OPENAI_API_KEY'), 'model': 'gpt-4o'})

# 连接到本地 SQLite 数据库（Chinook.sqlite）
vn.connect_to_sqlite('Chinook.sqlite')

# 从数据库读取 DDL 并训练，让模型了解真实表结构
df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
for ddl in df_ddl['sql'].to_list():
    vn.train(ddl=ddl)

# 你原有的训练示例（保留亦可，如不需要可删除）
vn.train(ddl="""
    CREATE TABLE IF NOT EXISTS my-table (
        id INT PRIMARY KEY,
        name VARCHAR(100),
        age INT
    )
""")

vn.train(documentation="Our business defines XYZ as ...")

vn.train(sql="SELECT name, age FROM my-table WHERE name = 'John Doe'")

print(vn.ask("What are the top 10 customers by sales?"))
