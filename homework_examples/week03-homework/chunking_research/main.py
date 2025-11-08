import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter, SentenceWindowNodeParser, MarkdownNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor


EVAL_TEMPLATE_STR = (
    "我们提供了一个标准答案和一个由模型生成的回答。请你判断模型生成的回答在语义上是否与标准答案一致、准确且完整。\n"
    "请只回答 '是' 或 '否'。\n"
    "\n"
    "标准答案：\n"
    "----------\n"
    "{ground_truth}\n"
    "----------\n"
    "\n"
    "模型生成的回答：\n"
    "----------\n"
    "{generated_answer}\n"
    "----------\n"
)
eval_template = PromptTemplate(template=EVAL_TEMPLATE_STR)


def evaluate_splitter(splitter, documents, question, ground_truth, splitter_name):
    """
    评估不同的分割器对检索和生成质量的影响。

    Args:
        splitter: 用于切分文档的 LlamaIndex 分割器对象。
        documents: 待处理的文档列表。
        question: 用于查询的问题字符串。
        ground_truth: 问题的标准答案。
        splitter_name: 分割器的名称，用于在结果中标识。
    """
    print(f"--- 开始评估: {splitter_name} ---")

    # 1. 使用指定的分割器切分文档
    nodes = splitter.get_nodes_from_documents(documents)
    
    # 2. 建立索引
    index = VectorStoreIndex(nodes)

    # 3. 创建查询引擎
    # 特别处理 SentenceWindowNodeParser，因为它需要一个后处理器
    if isinstance(splitter, SentenceWindowNodeParser):
        # 注意：句子窗口切片需要特殊的后处理器
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")]
        )
    else:
        query_engine = index.as_query_engine(similarity_top_k=5)

    # 4. 检索上下文
    retrieved_nodes = query_engine.retrieve(question)
    retrieved_context = "\n\n".join([node.get_content() for node in retrieved_nodes])

    # 5. 评估：检索到的上下文是否包含答案
    # 使用标准答案的前15个字符作为关键信息进行检查
    context_contains_answer = "是" if ground_truth[:15] in retrieved_context else "否"

    # 6. 评估：LLM 生成的回答是否准确完整
    response = query_engine.query(question)
    generated_answer = str(response)
    # 使用 LLM 作为评判者来评估答案的准确性
    eval_response = Settings.llm.predict(
        eval_template, 
        ground_truth=ground_truth, 
        generated_answer=generated_answer
    )
    answer_is_accurate = "是" if "是" in eval_response else "否"

    # 7. 评估：上下文冗余程度（人工评分）
    print("\n检索到的上下文：")
    print(retrieved_context)
    redundancy_score = input(f"请为【{splitter_name}】的上下文冗余程度打分 (1-5, 1表示最不冗余, 5表示最冗余): ")
    while redundancy_score not in ['1', '2', '3', '4', '5']:
        redundancy_score = input("无效输入。请输入1到5之间的整数: ")

    # 8. 存储结果
    # 使用函数属性来跨多次调用存储结果
    if not hasattr(evaluate_splitter, "results"):
        evaluate_splitter.results = []
    
    evaluate_splitter.results.append({
        "分割器": splitter_name,
        "上下文包含答案": context_contains_answer,
        "回答准确": answer_is_accurate,
        "上下文冗余度(1-5)": int(redundancy_score),
        "生成回答": generated_answer.strip().replace("\n", " ")[:100] + "..."
    })

    print(f"--- 完成评估: {splitter_name} ---\n")


def print_results_table():
    """打印所有评估结果的汇总表格。"""
    if hasattr(evaluate_splitter, "results") and evaluate_splitter.results:
        print("\n--- 最终评估结果对比 ---")
        df = pd.DataFrame(evaluate_splitter.results)
        print(df.to_markdown(index=False))
    else:
        print("没有可供显示的评估结果。")


def main():
    """主函数"""
    # 作业的入口写在这里。你可以就写这个文件，或者扩展多个文件，但是执行入口留在这里。
    # 在根目录可以通过python -m chunking_research.main 运行
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    print("loaded api key")
    
    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        is_chat_model=True
    )

    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,
        embed_input_length=8192
    )
    
    documents = SimpleDirectoryReader("data").load_data()
    
    # 句子切片
    sentence_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    
    # 定义具体的问题和标准答案，这里以量子计算为例
    question = "量子计算的基本原理是什么？"
    ground_truth = "量子计算基于量子力学原理，与传统的经典计算有着本质的区别。在经典计算机中，信息的基本单位是比特（Bit），它只有两种状态，即 0 或 1，就像普通的开关，要么开，要么关。而量子计算的基本信息单元是量子比特（Qubit），它具有独特的量子特性，不仅可以处于 0 或 1 的状态，还可以处于这两种状态的叠加态。这意味着一个量子比特能够同时表示 0 和 1，从而使量子计算机在同一时刻能够处理多种信息，具备了并行计算的能力。"
    
    evaluate_splitter(sentence_splitter, documents, question, ground_truth, "Sentence")
    
    # Token 切片
    splitter = TokenTextSplitter(
        chunk_size=128,
        chunk_overlap=4,
        separator="\n"
    )
    
    evaluate_splitter(splitter, documents, question, ground_truth, "Token")
    
    # 句子窗口切片
    sentence_window_splitter = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )

    evaluate_splitter(sentence_window_splitter, documents, question, ground_truth, "Sentence Window")
    
    
    markdown_splitter = MarkdownNodeParser()
    evaluate_splitter(markdown_splitter, documents, question, ground_truth, "Markdown")
    
    print_results_table()
    

if __name__ == "__main__":
    main()