# 第三周作业 Part 1

## 任务
1. 探索 LlamaIndex 中的句子切片检索（Sentence Window Retrieval）及其参数影响分析
2. 为 LlamaIndex 构建 OCR 图像文本加载器：基于 paddle-OCR 的多模态数据接入

## 作业一: 探索 LlamaIndex 中的句子切片检索及其参数影响分析
### 作业目标
1. 理解 LlamaIndex 框架中“句子切片”的核心思想与实现机制。
2. 实践使用 LlamaIndex 构建基于句子窗口的检索增强生成（RAG）系统。
3. 对比分析不同参数设置对检索效果和生成质量的影响。
4. 培养对文本分块策略与上下文保留之间权衡的理解。
### 技术背景简介
句子切片是一种特殊的文本分块策略：将文档按句子级别切分，但在检索时返回包含匹配句子的“上下文窗口”（即前后若干句子），从而在精确匹配与上下文完整性之间取得平衡。
### 作业任务与步骤
#### 环境搭建与数据准备
安装必要库：
```
pip install llama-index
pip install llama-index-core
pip install llama-index-llms-openai-like
pip install llama-index-llms-dashscope
pip install llama-index-embeddings-dashscope
```
或者使用uv安装，请在pyproject.toml中添加依赖，然后运行`uv sync`安装。
#### 获取 API KEY
参考链接（https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.27727b085cwI5h&tab=api#/api/?type=model&url=2712195）
获取访问 qwen 模型所需的API KEY ，并保存到 “DASHSCOPE_API_KEY” 环境变量中。
#### 准备一份文本数据集（至少 3 篇长文档，每篇 >1000 字），可选来源：
   - Wikipedia 文章（如“量子计算”、“气候变化”）
   - PDF 学术论文摘要
   - 自定义写作文档（如产品说明书、小说节选）
   - 由大模型生成
要求：文档需包含复杂句式和段落结构，便于观察切片效果。
#### 设置llamaindex 使用 qwen 系列，兼容 OpenAI 接口的大模型和嵌入模型
```python
import os
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True
)

Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    embed_batch_size=6,
    embed_input_length=8192
)
```
#### 使用句子切片
句子切片示例代码
```python
# 句子切片
sentence_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)
evaluate_splitter(sentence_splitter, documents, question, ground_truth, "Sentence")
```
#### 尝试其他切片方式
##### Token 切片
```python
# Token 切片
splitter = TokenTextSplitter(
    chunk_size=32,
    chunk_overlap=4,
    separator="\n"
)
```
##### 句子窗口切片
```python
sentence_window_splitter = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)

# 注意：句子窗口切片需要特殊的后处理器
query_engine = index.as_query_engine(
    similarity_top_k=5,
    streaming=True,
    node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")]
)

evaluate_splitter(sentence_window_splitter, documents, question, ground_truth, "Sentence Window")
```
##### markdown 切片（选做）（需准备markdown格式的文档，可采用模型生成）
```python
markdown_splitter = MarkdownNodeParser()
evaluate_splitter(markdown_splitter, documents, question, ground_truth, "Markdown")
```
#### 参数对比实验
比较不同参数组合对**检索相关性**和**生成回答质量**的影响
- 检索到的上下文是否包含答案
- LLM 生成的回答是否准确完整
- 上下文冗余程度（主观评分 1–5）
- 制作对比表格或图表展示结果。
#### 报告撰写
请补充[chunking_research/report.md](chunking_research/report.md) 文件，内容包括：
- 哪些参数显著影响效果？为什么？
- chunk_overlap 过大或过小的利弊？
- 如何在“精确检索”与“上下文丰富性”之间权衡
#### 参考资料
- LlamaIndex 官方文档：https://docs.llamaindex.ai


## 作业二: 为 LlamaIndex 构建 OCR 图像文本加载器：基于 paddle-OCR的多模态数据接入
### 作业目标
- 理解 LlamaIndex 中 Document 与 BaseReader 的设计模式。
- 掌握使用 PaddlePaddle 的 paddle-OCR 模型从图像中提取文本内容。
- 实现一个自定义 **ImageOCRReader**，将含文字的图像（如截图、扫描件、海报等）转换为 LlamaIndex 可处理的 Document 对象。
- 提升对多模态数据处理和 RAG 系统扩展能力的理解。
### 技术背景
- LlamaIndex 支持通过自定义 Reader 加载多种格式的数据（PDF、HTML、图像等），最终统一为 Document 对象用于索引和检索。
- paddle-OCR 是百度 PaddleOCR 项目推出的高性能轻量级 OCR 系统，支持多语言、检测 + 识别 + 方向分类一体化。
•本作业要求你构建一个桥梁：将图像 → 文本 → Document → LlamaIndex 索引流程打通。
### 作业任务与步骤
#### 环境搭建与依赖安装
- 安装必要库：
```bash
# 您的机器安装的是CUDA 11，请运行以下命令安装
pip install "paddlepaddle-gpu<=2.6"
# 您的机器是CPU，请运行以下命令安装
pip install "paddlepaddle<=2.6"
# 安装PaddleOCR whl包
pip install "paddleocr<3.0"
```
或者使用uv安装，请在pyproject.toml中添加依赖，然后运行`uv sync`安装。

- 验证 PP-OCR 是否可用：
```bash
paddleocr ocr -i ./general_ocr_002.png --ocr_version PP-OCRv4
```
- 官方文档参考：https://www.paddleocr.ai/latest/version3.x/pipeline_usage/OCR.html#22-python
- 要求：能正确输出检测框与识别文本。

#### 设计并实现 ImageOCRReader
- 创建一个继承自 BaseReader 的类，实现图像到 Document 的转换。
- 要求功能：
```python
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import os
from typing import List, Union

class ImageOCRReader(BaseReader):
    """使用 PP-OCR v5 从图像中提取文本并返回 Document"""
    
    def __init__(self, lang='ch', use_gpu=False, **kwargs):
        """
        Args:
            lang: OCR 语言 ('ch', 'en', 'fr', etc.)
            use_gpu: 是否使用 GPU 加速
            **kwargs: 其他传递给 PaddleOCR 的参数
        """
    
    def load_data(self, file: Union[str, List[str]]) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表
        Args:
            file: 图像路径字符串 或 路径列表
        Returns:
            List[Document]
        """
        # 实现 OCR 提取逻辑
        # 将每张图的识别结果拼接成文本
        # 构造 Document 对象，附带元数据（如 image_path, ocr_confidence_avg）
```

#### PP-OCR 调用 Python 脚本参考
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,  # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
    use_doc_unwarping=False,  # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
    use_textline_orientation=False,  # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
)

# ocr = PaddleOCR(lang="en")  # 通过 lang 参数来使用英文模型
# ocr = PaddleOCR(ocr_version="PP-OCRv4")  # 通过 ocr_version 参数来使用 PP-OCR 其他版本
# ocr = PaddleOCR(device="gpu")  # 通过 device 参数使得在模型推理时使用 GPU

# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_server_det",
#     text_recognition_model_name="PP-OCRv5_server_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# )  # 更换 PP-OCRv5_server 模型

result = ocr.predict("./general_ocr_002.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```
#### Document 元数据建议字段：
- image_path: 原始图像路径
- ocr_model: "PP-OCRv5"
- language: 使用的语言
- num_text_blocks: 检测到的文本块数量
- avg_confidence: 平均识别置信度

#### 示例输出文本格式（可自行设计）：
```
Plain Text[Text Block 1] (conf: 0.98): 欢迎使用 PP-OCR[Text Block 2] (conf: 0.95): 日期：2025年4月5日...
```



#### 测试与集成
1. 准备至少 3 类图像：
- 扫描文档（清晰文本）
- 屏幕截图（含 UI 文字）
- 自然场景图（如路牌、广告牌，挑战性高）

2. 使用 ImageOCRReader 加载这些图像，生成 Document 列表。
3. 将 Document 输入 LlamaIndex 构建索引，并进行简单查询验证：
```python
reader = ImageOCRReader(lang='ch')
documents = reader.load_data(["img1.png", "img2.jpg"])

from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("图片中提到了什么日期？")
print(response)
```

#### 报告撰写
请补充[ocr_research/report.md](ocr_research/report.md) 文件，内容包括：
- 架构设计图：展示 ImageOCRReader 在 LlamaIndex 流程中的位置。
- 核心代码说明：关键函数与类的设计思路。
- OCR 效果评估：各类图像的识别准确率（人工评估）
- 错误案例分析（如倾斜、模糊、艺术字体）
- Document 封装合理性讨论：文本拼接方式是否合理？元数据设计是否有助于后续检索？
- 局限性与改进建议：如何保留空间结构（如表格）？是否可加入 layout analysis（如 PP-Structure）？
#### 附加挑战（可选）
- 支持批量处理图像目录（load_data_from_dir(dir_path)）
- 可视化 OCR 检测框（使用 OpenCV 画出边界框并保存）
- 支持 PDF 扫描件（每页转图像后 OCR）

#### 参考资料
- Llamahub 中的Dataloader插件：https://llamahub.ai/?tab=readers
- LlamaIndex 的图像插件代码：https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image
- PP-OCRv5 文档：https://www.paddleocr.ai/latest/version3.x/pipeline_usage/OCR.html#21

## 如何提交作业
请fork本仓库，然后在以下目录分别完成编码作业：
- [week03-homework/chunking_research](chunking_research)
- [week03-homework/ocr_research](ocr_research)

其中:
- main.py是作业的入口
- report.md是作业的报告


完成作业后，请在【极客时间】上提交你的fork仓库链接，精确到本周的目录，例如：
```
https://github.com/Blackoutta/ai-engineer-training/tree/main/week03-homework`
```