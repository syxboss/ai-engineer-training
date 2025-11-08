import os
import json
import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen3意图识别API",
    description="基于Qwen3模型的意图识别服务",
    version="1.0.0"
)

# 模型配置
MODEL_PATH = "./qwen3-intent-lora"
LABEL_MAPPING_PATH = os.path.join(MODEL_PATH, "label_mapping.json")
BASE_MODEL_NAME = "Qwen/Qwen3-8B"

# 全局变量
model = None
tokenizer = None
id2label = None

class IntentRequest(BaseModel):
    """意图识别请求模型"""
    text: str
    
    class Config:
        schema_extra = {
            "example": {
                "text": "我要退票"
            }
        }

class IntentResponse(BaseModel):
    """意图识别响应模型"""
    text: str
    intent: str
    confidence: float

def load_model_and_tokenizer():
    """
    加载模型和分词器
    """
    global model, tokenizer, id2label
    
    try:
        logger.info("开始加载模型和分词器...")
        
        # 检查模型文件是否存在
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型路径不存在: {MODEL_PATH}")
        
        if not os.path.exists(LABEL_MAPPING_PATH):
            raise FileNotFoundError(f"标签映射文件不存在: {LABEL_MAPPING_PATH}")
        
        # 加载标签映射
        with open(LABEL_MAPPING_PATH, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        id2label = label_mapping["id2label"]
        logger.info(f"加载标签映射完成，共 {len(id2label)} 个类别")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("分词器加载完成")
        
        # 加载基础模型
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=len(id2label),
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        logger.info("基础模型加载完成")
        
        # 加载LoRA适配器
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model.eval()
        logger.info("LoRA适配器加载完成")
        
        logger.info("模型和分词器加载成功!")
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    load_model_and_tokenizer()

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "Qwen3意图识别API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict - 意图识别",
            "health": "GET /health - 健康检查"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    if model is None or tokenizer is None or id2label is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "num_labels": len(id2label),
        "available_intents": list(id2label.values())
    }

@app.post("/predict", response_model=IntentResponse)
async def predict_intent(request: IntentRequest):
    """
    意图识别预测端点
    
    Args:
        request: 包含待识别文本的请求
        
    Returns:
        IntentResponse: 包含预测意图和置信度的响应
    """
    try:
        # 检查模型是否已加载
        if model is None or tokenizer is None or id2label is None:
            raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")
        
        # 输入验证
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="输入文本不能为空")
        
        text = request.text.strip()
        logger.info(f"处理意图识别请求: {text}")
        
        # 文本预处理
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        # 将输入移动到模型设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 模型推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 获取预测结果
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_id = torch.argmax(logits, dim=1).item()
            confidence = float(probabilities[0][predicted_class_id].cpu())
        
        # 获取预测的意图
        predicted_intent = id2label[str(predicted_class_id)]
        
        logger.info(f"预测结果: {predicted_intent}, 置信度: {confidence:.4f}")
        
        # 返回预测结果
        return IntentResponse(
            text=text,
            intent=predicted_intent,
            confidence=confidence
        )
    
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        logger.error(f"预测过程中出现错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.get("/intents")
async def get_available_intents():
    """获取所有可用的意图类别"""
    if id2label is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "intents": list(id2label.values()),
        "count": len(id2label)
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("Qwen3意图识别API服务")
    print("=" * 50)
    print("API服务启动中...")
    print("可用端点:")
    print("  - GET  /          : API信息")
    print("  - GET  /health    : 健康检查")
    print("  - POST /predict   : 意图识别")
    print("  - GET  /intents   : 获取可用意图")
    print("示例请求: POST /predict")
    print('  {"text": "我要退票"}')
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )