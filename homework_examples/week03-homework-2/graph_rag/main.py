import uvicorn
from fastapi import FastAPI
from .api import router as api_router
from . import config
from . import graph_builder


app = FastAPI(
    title="GraphRAG 多跳问答系统",
    description="一个融合了文档检索 (RAG) 和知识图谱 (KG) 的高级问答 API",
    version="1.0.0",
)


app.include_router(api_router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    # 在应用启动时可以执行一些检查或预热操作
    print("应用启动...")
    # 可以在这里预热模型，但我们的 query_engine 在导入时已经初始化了
    print("查询引擎已准备就绪。")


@app.get("/")
def read_root():
    return {
        "message": "欢迎使用 GraphRAG API. "
                   "请先运行 'python -m graph_rag.graph_builder' 来构建知识图谱, "
                   "然后访问 /docs 查看 API 文档."
    }


def main():
# 作业的入口写在这里。你可以就写这个文件，或者扩展多个文件，但是执行入口留在这里。
# 在根目录可以通过python -m graph_rag.main 运行
    print("启动 FastAPI 服务...")
    print("在启动服务前，请确保您已经运行了图谱构建脚本:")
    print("python -m graph_rag.graph_builder")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()