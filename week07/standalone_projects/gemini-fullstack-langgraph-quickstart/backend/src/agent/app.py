# mypy: 禁用错误代码 - "no-untyped-def,misc"
import pathlib
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

# 定义FastAPI应用实例
app = FastAPI()


def create_frontend_router(build_dir="../frontend/dist"):
    """
    创建一个路由器来服务React前端
    
    Args:
        build_dir: 相对于此文件的React构建目录路径
        
    Returns:
        服务前端的Starlette应用程序
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    # 检查构建目录是否存在且包含index.html文件
    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"警告: 在 {build_path} 找不到前端构建目录或目录不完整。服务前端可能会失败。"
        )
        # 如果构建未准备好，返回一个虚拟路由器
        from starlette.routing import Route

        async def dummy_frontend(request):
            return Response(
                "前端未构建。请在前端目录中运行 'npm run build'。",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    # 返回静态文件服务器，支持HTML模式（用于SPA路由）
    return StaticFiles(directory=build_path, html=True)


# 将前端挂载到/app路径下，以避免与LangGraph API路由冲突
app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)
