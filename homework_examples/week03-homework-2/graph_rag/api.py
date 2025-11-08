from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from . import query_engine


router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., example="星辰科技的最大股东是谁？")


class QueryResponse(BaseModel):
    final_answer: str
    reasoning_path: List[str]


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    接收多跳问答查询
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    try:
        result = query_engine.multi_hop_query(request.question)
        return result
    except Exception as e:
        # 打印详细错误信息到服务器日志，方便调试
        print(f"查询处理时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理查询时发生内部错误: {str(e)}")