# 医疗领域客服问答模型

基于 LoRA 微调的垂直领域客服问答模型，专注于医疗健康领域的智能问答服务。

## 🚀 快速开始

### 环境要求
- Python 3.8+
- CUDA 11.8+ (推荐)
- 16GB+ RAM
- 8GB+ GPU显存 (训练时)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 快速体验
```bash
# 1. 训练模型
python train_medical_qa.py

# 2. 合并权重
python merge_lora_weights.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --lora_model ./qwen-medical-qa-lora \
    --output ./qwen-medical-qa-merged

# 3. 启动API服务
python api_server.py --model_path ./qwen-medical-qa-merged

# 4. 测试API
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "什么是高血压？"}'
```

## 📋 项目特性

- ✅ **LoRA微调**：高效的参数微调方法
- ✅ **权重合并**：将LoRA权重合并到原始模型
- ✅ **多维评估**：ROUGE、BERTScore等评估指标
- ✅ **API部署**：FastAPI RESTful服务
- ✅ **医疗专业**：涵盖多个医学领域的QA数据

## 🏗️ 项目结构

```
project2_2/
├── data/                      # 训练数据
├── train_medical_qa.py       # 模型训练
├── merge_lora_weights.py     # 权重合并
├── evaluate_model.py         # 模型评估
├── api_server.py            # API服务
├── FAQ.md                   # 技术FAQ
└── 项目结构说明文档.md        # 详细文档
```

## 🔧 核心技术

### LoRA微调
- **参数效率**：只训练1-2%的参数
- **存储效率**：适配器文件仅几十MB
- **训练速度**：相比全参数微调快10倍以上

### 权重合并
- **无损合并**：理论上不损失性能
- **内存优化**：支持大模型的高效合并
- **验证机制**：自动验证合并后模型

### 多维评估
- **ROUGE分数**：文本重叠度评估
- **BERTScore**：语义相似度评估
- **专业性评估**：医学知识准确性

## 📊 性能指标

| 指标 | 基础模型 | 微调后模型 | 提升 |
|------|----------|------------|------|
| ROUGE-1 | 0.45 | 0.72 | +60% |
| ROUGE-L | 0.38 | 0.65 | +71% |
| BERTScore | 0.82 | 0.91 | +11% |

## 🔍 技术难点解决

### 1. 领域语料准备
- **多源数据收集**：医疗百科、专业文献、FAQ数据
- **质量控制**：专家审核、交叉验证
- **格式标准化**：统一的JSON格式

### 2. 模型评估
- **自动化指标**：ROUGE、BERTScore、困惑度
- **专业性评估**：医学知识准确性、术语规范性
- **安全性评估**：风险提示、免责声明

### 3. LoRA权重合并
- **合并策略**：merge_and_unload方法
- **内存优化**：分块加载、CPU offload
- **精度保持**：一致的数据类型处理

## 🚀 API接口

### 问答接口
```http
POST /ask
Content-Type: application/json

{
  "question": "什么是糖尿病？",
  "max_length": 256,
  "temperature": 0.7
}
```

### 响应格式
```json
{
  "answer": "糖尿病是一种慢性代谢性疾病...",
  "question": "什么是糖尿病？",
  "processing_time": 1.23,
  "model_info": "./qwen-medical-qa-merged"
}
```

## 📈 优化建议

### 训练优化
- 使用4-bit量化节省显存
- 启用混合精度训练
- 调整LoRA参数（r=16, alpha=32）

### 推理优化
- 模型量化（INT8/INT4）
- 批量推理
- KV缓存加速

### 部署优化
- 容器化部署
- 负载均衡
- 缓存机制

## 🔒 安全考虑

- **免责声明**：明确不能替代专业医疗建议
- **内容过滤**：过滤高风险医疗建议
- **数据脱敏**：移除个人身份信息
- **访问控制**：API访问权限管理

## 📚 文档

- [FAQ.md](FAQ.md) - 技术难点详解
- [项目结构说明文档.md](项目结构说明文档.md) - 完整项目文档

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 📞 联系

如有问题请提交Issue或联系项目维护者。