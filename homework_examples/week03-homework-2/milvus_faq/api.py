from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from . import index_manager


router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    score: float


@router.post("/query", response_model=list[QueryResponse])
async def query_faq(request: QueryRequest):
    """
    接收用户问题并返回最相关的FAQ条目。
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    print(f"收到查询: {request.question}")
    query_engine = index_manager.get_query_engine()
    response = query_engine.query(request.question)

    if not response.source_nodes:
        return []

    results = []
    for node in response.source_nodes:
        # 从节点文本中解析出原始问题和答案
        text_parts = node.get_text().split('\n答案: ')
        original_question = text_parts[0].replace('问题: ', '')
        answer = text_parts[1] if len(text_parts) > 1 else "答案未找到"
        
        results.append(
            QueryResponse(
                question=original_question,
                answer=answer,
                score=node.get_score() or 0.0
            )
        )
    
    return results


@router.post("/update-index")
async def update_faq_index():
    """
    触发知识库的热更新。
    系统将从 data/faqs.csv 重新加载并建立索引。
    """
    try:
        result = index_manager.update_index()
        return {"status": "success", "message": result["message"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"索引更新失败: {str(e)}")
