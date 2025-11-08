import uvicorn
from fastapi import FastAPI
from .api import router as api_router
from . import config # 确保配置在启动时被加载和检查


app = FastAPI(
    title="Milvus FAQ 检索系统",
    description="一个基于 LlamaIndex 和 Milvus 的 FAQ 问答 API",
    version="1.0.0",
)


app.include_router(api_router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": "欢迎使用 Milvus FAQ 检索系统 API. 请访问 /docs 查看详情."}


def main():
# 作业的入口写在这里。你可以就写这个文件，或者扩展多个文件，但是执行入口留在这里。
# 在根目录可以通过python -m milvus_faq.main 运行
    print("启动 FastAPI 服务...")
    # 打印一些启动信息
    print(f"数据文件路径: {config.FAQ_FILE}")
    print(f"Milvus Lite 数据库路径: {config.MILVUS_URI}")
    print(f"嵌入模型: {config.EMBED_MODEL.model_name}")
    
    # 运行 FastAPI 应用
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()