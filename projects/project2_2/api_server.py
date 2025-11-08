from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import os
import logging
from typing import Optional
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 请求和响应模型
class QuestionRequest(BaseModel):
    question: str
    max_length: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class AnswerResponse(BaseModel):
    answer: str
    question: str
    processing_time: float
    model_info: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

# 全局变量
app = FastAPI(
    title="医疗QA API服务",
    description="基于LoRA微调的医疗领域问答API",
    version="1.0.0"
)

model = None
tokenizer = None
device = None
model_path = None

def load_model(model_path: str):
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径
    """
    global model, tokenizer, device
    
    try:
        logger.info(f"开始加载模型: {model_path}")
        
        # 检查CUDA可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")
        
        # 加载分词器
        logger.info("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        logger.info("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else "cpu",
            trust_remote_code=True
        )
        
        logger.info("模型加载完成")
        return True
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return False

def generate_answer(question: str, max_length: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
    """
    生成问题的答案
    
    Args:
        question: 输入问题
        max_length: 最大生成长度
        temperature: 温度参数
        top_p: top_p参数
        
    Returns:
        生成的答案
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 构建输入提示
        prompt = f"<|im_start|>system\n你是一个专业的医疗助手，请根据医学知识准确回答用户的健康相关问题。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手回答部分
        if "<|im_start|>assistant\n" in generated_text:
            answer = generated_text.split("<|im_start|>assistant\n")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        return answer
        
    except Exception as e:
        logger.error(f"生成答案时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成答案失败: {str(e)}")

@app.get("/", response_model=dict)
async def root():
    """根路径"""
    return {
        "message": "医疗QA API服务",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        device=str(device) if device else "unknown"
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    问答接口
    
    Args:
        request: 问题请求
        
    Returns:
        答案响应
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    start_time = time.time()
    
    try:
        # 生成答案
        answer = generate_answer(
            question=request.question,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        processing_time = time.time() - start_time
        
        return AnswerResponse(
            answer=answer,
            question=request.question,
            processing_time=processing_time,
            model_info=model_path or "unknown"
        )
        
    except Exception as e:
        logger.error(f"处理问题时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload_model")
async def reload_model(new_model_path: str):
    """
    重新加载模型
    
    Args:
        new_model_path: 新模型路径
    """
    global model_path
    
    if not os.path.exists(new_model_path):
        raise HTTPException(status_code=400, detail=f"模型路径不存在: {new_model_path}")
    
    success = load_model(new_model_path)
    
    if success:
        model_path = new_model_path
        return {"message": "模型重新加载成功", "model_path": new_model_path}
    else:
        raise HTTPException(status_code=500, detail="模型重新加载失败")

@app.get("/model_info")
async def get_model_info():
    """获取模型信息"""
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_path": model_path,
            "device": str(device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_type": model.config.model_type if hasattr(model, 'config') else "unknown"
        }
        
    except Exception as e:
        logger.error(f"获取模型信息时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="启动医疗QA API服务")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="服务器主机")
    parser.add_argument("--port", type=int, default=8000, 
                       help="服务器端口")
    parser.add_argument("--reload", action="store_true", 
                       help="开发模式，自动重载")
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        return
    
    # 加载模型
    global model_path
    model_path = args.model_path
    
    success = load_model(args.model_path)
    if not success:
        logger.error("模型加载失败，退出程序")
        return
    
    # 启动服务器
    logger.info(f"启动API服务器: http://{args.host}:{args.port}")
    logger.info(f"API文档: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()