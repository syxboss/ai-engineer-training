# 混合RAG系统 - 模块化重构版本

## 项目概述

这是一个经过完全重构的混合RAG（检索增强生成）系统，采用模块化设计，提供了更好的代码组织、可维护性和扩展性。

## 系统特性

### 🏗️ 模块化架构
- **配置管理**: 集中化的配置参数管理
- **数据模型**: 完整的数据结构定义
- **文本嵌入**: 支持通义千问和本地模型
- **多模态检索**: 向量检索 + 关键词检索 + 图谱推理
- **知识图谱**: 基于Neo4j的图谱推理
- **错误处理**: 完善的错误传播防护机制

### 🚀 核心功能
1. **中文优化**: 使用jieba分词和通义千问嵌入
2. **混合检索**: 结合多种检索策略
3. **图谱推理**: 支持多跳关系查询
4. **配置灵活**: 支持环境变量和动态配置
5. **监控完善**: 系统指标和性能监控

## 项目结构

```
project3_2/
├── __init__.py              # 包初始化文件
├── config.py                # 配置管理模块
├── models.py                # 数据模型定义
├── embedding.py             # 文本嵌入服务
├── retrieval.py             # 检索功能模块
├── graph_reasoning.py       # 图谱推理模块
├── hybrid_rag_system.py     # 主系统集成
├── demo.py                  # 演示程序
├── improved_hybrid_rag.py   # 原始文件（保留）
└── README.md               # 项目说明
```

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export DASHSCOPE_API_KEY="your-api-key"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
```

### 2. 基本使用

```python
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM

# 导入重构后的模块
from hybrid_rag_system import ImprovedHybridRAGSystem

# 初始化组件
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
llm_json = OpenAILLM(model_name="gpt-4o-mini", model_params={"response_format": {"type": "json_object"}})
llm_text = OpenAILLM(model_name="gpt-4o-mini")

# 创建系统
system = ImprovedHybridRAGSystem(driver, llm_json, llm_text, use_qwen_embedding=True)

# 加载数据
raw_text = system.load_data_from_file("data.txt")
documents = await system.process_text_to_documents(raw_text)
system.add_documents(documents)

# 构建图谱
relationships = await system.build_knowledge_graph(raw_text)

# 问答
result = await system.multi_hop_qa("你的问题")
print(result.answer)
```

### 3. 运行演示

```bash
# 运行演示程序
python demo.py
```

## 测试结果

✅ **所有模块导入测试通过**
- 配置模块导入成功
- 数据模型导入成功  
- 嵌入模块导入成功
- 检索模块导入成功（关键词提取功能正常）
- 图谱推理模块导入成功
- 主系统模块导入成功

## 重构成果

### 原始文件 vs 重构后
- **原始**: 单个724行的大文件
- **重构**: 8个模块化文件，职责清晰
- **可维护性**: 大幅提升
- **可扩展性**: 支持独立模块开发
- **可测试性**: 每个模块可独立测试

### 主要改进
1. **模块化设计**: 按功能拆分为独立模块
2. **配置集中化**: 统一的配置管理
3. **错误处理**: 完善的异常处理机制
4. **代码注释**: 详细的文档和注释
5. **导入兼容**: 支持相对和绝对导入

## 版本信息

- **版本**: 2.0.0
- **状态**: ✅ 重构完成，测试通过
- **兼容性**: 保持原有功能的同时提升架构

# 融合RAG与图谱推理的多跳问答系统

## 项目概述

本项目实现了一个融合文档检索（RAG）、图谱推理与LLM生成的多跳问答系统，专门用于处理复杂的企业关系查询，如"A公司的最大股东是谁？"等问题。

## 核心技术特点

### 1. 多源信息融合
- **向量检索**：基于语义相似度的文档检索
- **关键词匹配**：基于文本匹配的精确检索  
- **图谱推理**：基于知识图谱的多跳推理

### 2. 联合评分机制
- 结合向量相似度、关键词匹配度和图谱置信度
- 多源信息加权融合
- 结果一致性验证

### 3. 错误传播防护
- 置信度阈值过滤
- 多路径验证
- 结果交叉验证
- 异常检测和降级处理

## 技术难点解决方案

### 如何将RAG与图谱推理融合？
1. **多源信息整合**：同时利用文档检索和图谱推理的结果
2. **联合评分机制**：通过加权融合不同信息源的置信度
3. **一致性验证**：检查不同信息源结果的一致性

### 如何设计联合评分机制？
1. **分层评分**：分别计算向量、关键词、图谱三个维度的分数
2. **加权融合**：根据任务特点设置不同权重
3. **一致性奖励**：对一致的结果给予额外加分
4. **置信度归一化**：确保最终分数在合理范围内

### 如何防止错误传播？
1. **置信度阈值**：过滤低置信度的结果
2. **多路径验证**：通过多个推理路径验证结果
3. **异常检测**：识别不一致或异常的结果
4. **降级处理**：对可疑结果降低置信度
5. **警告机制**：向用户提示潜在的不确定性

## 系统架构

```
用户问题
    ↓
实体提取 → 并行检索 → 联合评分 → 错误防护 → 答案生成
    ↓         ↓         ↓         ↓         ↓
查询解析   向量检索   多源融合   置信度过滤   LLM生成
    ↓      关键词检索   一致性检查   异常检测    结果输出
实体识别    图谱推理    权重计算    降级处理
```

## 安装和使用

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动Neo4j数据库
```bash
# 确保Neo4j运行在localhost:7687
# 用户名：neo4j，密码：password
```

### 3. 运行演示

#### 方式一：运行演示程序（推荐）
```bash
python demo.py
```

演示程序提供三种模式：
1. **完整系统演示** - 展示完整的RAG系统功能
2. **组件功能测试** - 测试各个模块的独立功能
3. **两者都运行** - 先测试组件，再运行完整演示

#### 方式二：直接使用系统
```bash
python hybrid_rag_system.py
```

#### 方式三：作为包导入使用
```python
from hybrid_rag_system import ImprovedHybridRAGSystem
from config import get_config

# 创建系统实例
config = get_config()
system = ImprovedHybridRAGSystem(config)
```

## 核心类说明

### HybridRAGSystem
主要的系统类，包含以下核心方法：

- `add_documents()`: 添加文档到检索库
- `vector_search()`: 向量语义检索
- `keyword_search()`: 关键词精确匹配
- `graph_reasoning()`: 图谱多跳推理
- `calculate_joint_score()`: 联合评分机制
- `error_propagation_guard()`: 错误传播防护
- `multi_hop_qa()`: 多跳问答主流程

### 数据结构

- `Entity`: 实体对象（公司、人员等）
- `Relationship`: 关系对象（控股、投资等）
- `Document`: 文档对象（包含向量嵌入）
- `RetrievalResult`: 检索结果
- `GraphResult`: 图谱推理结果

## 配置参数

```python
config = {
    'confidence_threshold': 0.6,      # 置信度阈值
    'max_retrieval_results': 5,       # 最大检索结果数
    'vector_weight': 0.4,             # 向量检索权重
    'keyword_weight': 0.3,            # 关键词检索权重  
    'graph_weight': 0.3,              # 图谱推理权重
    'error_propagation_threshold': 0.5 # 错误传播阈值
}
```

## 示例查询

- "A公司的最大股东是谁？"
- "B投资公司控制哪些公司？"
- "A公司有多少层级的控股关系？"

## 输出示例

```json
{
    "question": "A公司的最大股东是谁？",
    "entities": [{"name": "A公司", "type": "Company", "confidence": 0.9}],
    "retrieval_results": {
        "vector": [...],
        "keyword": [...]
    },
    "graph_reasoning": {
        "relationships": [...],
        "reasoning_paths": [...],
        "confidence": 0.85
    },
    "scoring": {
        "joint_score": 0.82,
        "confidence_level": "high"
    },
    "final_answer": "根据图谱推理和文档检索，B投资公司是A公司的最大股东..."
}
```

## 扩展建议

1. **支持更多实体类型**：人员、机构、产品等
2. **增强图谱推理**：支持更复杂的关系类型
3. **优化向量模型**：使用领域特定的嵌入模型
4. **添加缓存机制**：提高查询效率
5. **支持实时更新**：动态更新文档和图谱

## 注意事项

1. 确保Neo4j数据库正常运行
2. 首次运行会下载向量模型，需要网络连接
3. 大量文档可能需要较多内存
4. 建议根据具体场景调整配置参数