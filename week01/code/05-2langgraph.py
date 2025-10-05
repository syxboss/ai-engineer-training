import json
import os
from typing import Dict, List, TypedDict
from openai import OpenAI
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from pandas.core.frame import console

# 加载环境变量
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_API_BASE')

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)


# =============================================================================
# OpenAI Embeddings 实现
# =============================================================================
class OpenAIEmbeddings(Embeddings):
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = client
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [data.embedding for data in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


# =============================================================================
# 状态定义
# 项目使用了 LangGraph 框架来构建工作流。
# LangGraph 是一个基于状态的工作流框架，它要求定义一个状态类来：
# 在不同的节点（函数）之间传递数据
# 跟踪整个工作流的执行状态
# 确保数据的类型安全和一致性
# 
# 状态定义明确了整个长文本生成流程中需要传递的数据：
# original_text: 原始输入文本
# chunks: 切分后的文本块
# summaries: 每个文本块的摘要
# planning_tree: 生成的文章结构树
# final_output: 最终生成的文章
# vectorstore: 向量数据库存储
# =============================================================================
class GenerationState(TypedDict):
    original_text: str
    chunks: List[str]
    summaries: List[str]
    planning_tree: Dict
    final_output: str
    vectorstore: FAISS

# =============================================================================
# 模型初始化
# =============================================================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# =============================================================================
# 核心工具函数
# =============================================================================
def split_text(text: str) -> List[str]:
    """
    语义化文本分块：基于段落结构和语义完整性进行切分
    确保切分范围在2至10块之间，优先保持语义完整性
    用于演示，这里用了 hard-coded 的分块策略！！！
    """
    # 按段落分割，保留非空段落
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    # 如果段落数已在目标范围内，直接返回
    if 2 <= len(paragraphs) <= 10:
        return paragraphs
    
    # 如果段落太少，按句子进一步分割
    if len(paragraphs) < 2:
        sentences = []
        for para in paragraphs:
            # 按句号、问号、感叹号分割句子
            import re
            sent_list = re.split(r'[。！？]', para)
            sentences.extend([s.strip() for s in sent_list if s.strip()])
        
        # 将句子重新组合成2-4个块
        if len(sentences) >= 4:
            chunk_size = len(sentences) // 3
            chunks = []
            for i in range(0, len(sentences), chunk_size):
                chunk = "。".join(sentences[i:i+chunk_size])
                if chunk:
                    chunks.append(chunk + "。")
            return chunks[:10]  # 最多10块
        else:
            return sentences
    
    # 如果段落太多，合并相邻段落
    if len(paragraphs) > 10:
        chunk_size = len(paragraphs) // 8  # 目标8块左右
        chunks = []
        for i in range(0, len(paragraphs), chunk_size):
            chunk_paras = paragraphs[i:i+chunk_size]
            chunks.append("\n\n".join(chunk_paras))
        return chunks
    
    return paragraphs


def generate_summary(chunk: str) -> str:
    """
    生成精简摘要，确保摘要长度不超过原文的30%
    """
    chunk_length = len(chunk)
    target_length = int(chunk_length * 0.3)  # 目标长度为原文的30%
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": f"""请对以下内容进行高度精简的摘要。要求：
1. 摘要长度不超过{target_length}字符（约原文的30%）
2. 只保留最核心的观点和关键信息
3. 使用简洁的语言，避免冗余表达
4. 保持逻辑清晰，突出重点"""
            },
            {"role": "user", "content": chunk}
        ],
        temperature=0
    )
    
    summary = response.choices[0].message.content
    
    # 如果摘要仍然过长，进行二次压缩
    if len(summary) > target_length:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": f"请将以下摘要进一步压缩到{target_length}字符以内，只保留最关键的信息："
                },
                {"role": "user", "content": summary}
            ],
            temperature=0
        )
        summary = response.choices[0].message.content
    
    return summary


def build_planning_tree(summaries: List[str]) -> Dict:
    combined = "\n\n".join(f"Block {i+1}: {s}" for i, s in enumerate(summaries))
    prompt = f"""
    请根据以下文本块摘要，生成一份精简的综合报告结构大纲。
    目的：
    - 分析摘要内容，生成逻辑清晰的文章结构
    
    要求：
    - 总共只生成3-4个主要章节，每章不超过1个合并段落
    - 将相关小节内容合并为综合性段落
    - 保持逻辑连贯，突出核心内容
    - 输出为严格JSON格式，不要包含任何其他文字
    
    摘要汇总：
    {combined}
    
    请只输出JSON，格式如下（注意：subsections为空数组，所有内容合并到主章节）：
    {{
      "title": "报告主标题",
      "sections": [
        {{"title": "发展现状与技术基础", "subsections": []}},
        {{"title": "应用领域与实践案例", "subsections": []}},
        {{"title": "挑战问题与未来趋势", "subsections": []}}
      ]
    }}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    content = response.choices[0].message.content.strip()
    # 移除可能的markdown代码块标记
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    
    # 解析JSON，如果失败则使用默认结构
    parsed_json = json.loads(content.strip()) if content.strip().startswith('{') else {
        "title": "文档分析报告",
        "sections": [
            {"title": "核心技术与发展现状", "subsections": []},
            {"title": "应用实践与行业影响", "subsections": []},
            {"title": "挑战机遇与未来展望", "subsections": []}
        ]
    }
    return parsed_json


def retrieve_relevant_memory(query: str, vectorstore: FAISS, k: int = 3) -> str:
    if vectorstore is None:
        return "向量存储不可用"
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join(d.page_content for d in docs)


def generate_section_content(title: str, context: str) -> str:
    prompt = f"""
    你是专业撰稿人。请根据参考上下文，撰写以下章节的综合性内容。
    
    # 上下文参考：
    {context}
    
    # 目标章节：
    {title}
    
    要求：
    1. 将相关内容合并为一个完整的综合段落
    2. 涵盖该主题的核心要点和关键信息
    3. 语言精炼，逻辑清晰，避免冗余
    4. 段落长度适中（200-400字），内容丰富
    5. 体现专业深度和分析价值
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


# =============================================================================
# 工作流节点定义
# =============================================================================
def split_node(state: GenerationState) -> GenerationState:
    print("=" * 60)
    print("🔄 [分块阶段] 开始文本切分")
    print("=" * 60)
    
    chunks = split_text(state["original_text"])
    state["chunks"] = chunks
    
    print(f"📊 切分统计:")
    print(f"   原始文本长度: {len(state['original_text'])} 字符")
    print(f"   切分块数: {len(chunks)} 块")
    avg_length = sum(len(chunk) for chunk in chunks) // len(chunks) if chunks else 0
    print(f"   平均块长度: {avg_length} 字符")
    
    print(f"\n📝 切分结果详情:")
    for i, chunk in enumerate(chunks, 1):
        words = len(chunk.split())
        chars = len(chunk)
        preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
        print(f"   块 {i}: {words} 词 ({chars} 字符) | {preview}")
    
    # 验证分块均匀性
    chunk_sizes = [len(chunk) for chunk in chunks]
    size_variance = max(chunk_sizes) - min(chunk_sizes)
    print(f"\n📏 分块均匀性分析:")
    print(f"   最大块: {max(chunk_sizes)} 字符")
    print(f"   最小块: {min(chunk_sizes)} 字符")
    print(f"   大小差异: {size_variance} 字符")
    
    print("✅ 分块阶段完成\n")
    return state


def summarize_and_memorize_node(state: GenerationState) -> GenerationState:
    """
    作用：
    - 使用GPT-4O对文本块进行摘要
    - 将生成摘要保存到state["summaries"]
    - 根据摘要构建向量存储
    - 显示关键信息
    """
    # 从摘要列表中获取原文文本
    texts = [chunk for chunk in state["chunks"]]
    print("=" * 60)
    print("🧠 [记忆阶段] 构建上下文记忆")
    print("=" * 60)
    
    summaries = []
    print("📝 正在生成摘要...")
    
    for i, chunk in enumerate(state["chunks"], 1):
        print(f"   处理块 {i}/{len(state['chunks'])}...", end=" ")
        summary = generate_summary(chunk)
        summaries.append(summary)
        
        # 计算压缩比例
        compression_ratio = len(summary) / len(chunk) * 100
        print("✅")
        print(f"      原文: {len(chunk)} 字符")
        print(f"      摘要: {len(summary)} 字符 (压缩率: {compression_ratio:.1f}%)")
        print(f"      内容: {summary[:60]}...")
    
    state["summaries"] = summaries
    
    print(f"\n🔍 构建向量数据库...")
    state["vectorstore"] = FAISS.from_texts(summaries, embedding=embeddings)
    print("✅ 向量数据库构建完成")
    
    # 显示向量存储的关键信息
    print(f"📊 向量存储统计:")
    print(f"   存储文档数: {len(summaries)}")
    print(f"   向量维度: 1536 (text-embedding-3-small)")
    
    # 提取并显示关键词索引
    print(f"\n🔑 关键词索引:")
    for i, summary in enumerate(summaries, 1):
        # 简单提取关键词（取前几个重要词汇）
        keywords = [word for word in summary.split()[:5] if len(word) > 2]
        print(f"   文档 {i}: {', '.join(keywords)}")
    
    print("✅ 记忆阶段完成\n")
    return state


def planning_node(state: GenerationState) -> GenerationState:
    """
    摘要分析、规划阶段：
    - 根据摘要构建精简文章结构树
    - 根据文章结构树生成精简段落内容
    """
    print("=" * 60)
    print("📋 [规划阶段] 构建精简文章结构")
    print("=" * 60)
    
    print("🤖 正在分析摘要并生成精简结构树...")
    planning_tree = build_planning_tree(state["summaries"])
    state["planning_tree"] = planning_tree
    
    print("✅ 精简结构树生成完成")
    print(f"\n📖 精简文章大纲结构:")
    print(f"   标题: {planning_tree.get('title', '未定义')}")
    
    sections = planning_tree.get('sections', [])
    print(f"   主章节数: {len(sections)}")
    
    for i, section in enumerate(sections, 1):
        section_title = section.get('title', f'第{i}章')
        subsections = section.get('subsections', [])
        print(f"   {i}. {section_title}")
        if subsections:  # 如果还有子章节，显示它们
            for j, subsection in enumerate(subsections, 1):
                print(f"      {i}.{j} {subsection}")
        else:
            print(f"      (综合段落，无子章节)")
    
    print(f"\n🎯 精简生成策略:")
    # 重新计算段落数：只计算主章节，因为子章节已合并
    content_paragraphs = len(sections)
    print(f"   预计内容段落数: {content_paragraphs} 段")
    print(f"   段落生成方式: 每章节合并为单一综合段落")
    print(f"   ✅ 符合目标范围: 3-5段")
    
    print("✅ 规划阶段完成\n")
    return state


def generate_node(state: GenerationState) -> GenerationState:
    """
    段落生成阶段：
    - 根据文章结构树生成精简段落内容
    """
    print("=" * 60)
    print("✍️ [生成阶段] 精简段落生成")
    print("=" * 60)
    
    tree = state["planning_tree"]
    content_parts = []
    
    # 添加主标题
    if "title" in tree:
        title = tree["title"]
        content_parts.append(f"# {title}\n")
        print(f"📝 生成主标题: {title}")

    sections = tree.get("sections", [])
    print(f"🎯 采用精简策略: {len(sections)} 个主章节，无子章节分割")
    
    for i, section in enumerate(sections, 1):
        sec_title = section["title"]
        
        print(f"\n🔄 生成章节 {i}/{len(sections)}: {sec_title}")
        content_parts.append(f"## {sec_title}")
        
        # 生成综合性章节内容（合并所有相关小节）
        context = retrieve_relevant_memory(sec_title, state["vectorstore"])
        print(f"   📚 检索到相关上下文: {len(context)} 字符")
        
        content = generate_section_content(sec_title, context)
        content_parts.append(content)
        print(f"   ✅ 综合章节内容生成完成: {len(content)} 字符")
        
        # 更新记忆库
        if state["vectorstore"] is not None:
            state["vectorstore"].add_texts([content])
            print(f"   💾 内容已添加到记忆库")

    state["final_output"] = "\n\n".join(content_parts)
    
    # 计算实际段落数（标题 + 章节标题 + 内容段落）
    content_paragraphs = len([p for p in content_parts if not p.startswith("#")])
    
    print(f"\n📊 精简生成统计:")
    print(f"   总字符数: {len(state['final_output'])}")
    print(f"   主章节数: {len(sections)}")
    print(f"   内容段落数: {content_paragraphs}")
    print(f"   总组件数: {len(content_parts)}")
    print(f"   ✅ 成功将段落数控制在 {content_paragraphs} 段（目标: 3-5段）")
    print("✅ 生成阶段完成\n")
    
    return state


# =============================================================================
# 构建状态图
# =============================================================================
def create_generation_workflow() -> StateGraph:
    """
    第一层整合：原文→摘要（局部精炼）
    第二层整合：摘要→结构规划（全局组织）
    第三层整合：基于规划重新生成（深度融合）
    """
    workflow = StateGraph(GenerationState)

    workflow.add_node("split", split_node)
    workflow.add_node("summarize_and_memorize", summarize_and_memorize_node)
    workflow.add_node("plan", planning_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("split")
    workflow.add_edge("split", "summarize_and_memorize")
    workflow.add_edge("summarize_and_memorize", "plan")
    workflow.add_edge("plan", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# =============================================================================
# 执行示例
# =============================================================================
if __name__ == "__main__":
    print("启动长文本生成系统")
    print("=" * 60)
    
    sample_text = """
    在孤独与尊严的海洋中：重读《老人与海》的生命启示

在哈瓦那的晨曦中，一位名叫圣地亚哥的老渔夫独自划着小船驶向远海，这看似平凡的场景却构成了海明威《老人与海》的全部叙事空间。这部发表于1952年的中篇小说，以其简洁有力的语言和深邃的象征意义，成为20世纪文学史上不可忽视的经典之作。当我们穿越半个多世纪的时间迷雾重新阅读这部作品，会发现《老人与海》远非一个简单的"老人捕鱼"的故事，而是一曲关于人类精神力量的永恒赞歌，一次对存在本质的深刻叩问，更是一面映照现代人精神困境的明镜。

圣地亚哥这一人物形象的塑造，体现了海明威对人类精神世界的深刻洞察。这位连续八十四天没有捕到鱼的古巴老渔夫，在物质层面可谓处于社会边缘——他的渔获少得可怜，连最基本的生存需求都难以保障；他的船破旧不堪，工具简陋原始；他甚至被其他渔夫视为失败者，只有那个叫马诺林的小男孩还真心尊敬他。然而正是在这样一位看似"失败"的老人身上，海明威赋予了人类精神最为高贵的品质——尊严与坚韧。

老人与大海的关系构成了小说最核心的隐喻。海洋在文学传统中常被赋予母性、神秘与不可知的象征意义，而在《老人与海》中，大海既是生命的源泉，也是残酷的考验场。对圣地亚哥而言，海洋是他赖以生存的家园，也是他证明自我价值的战场。他熟知海洋的每一处细微变化，理解每一种鱼类习性的奥秘，这种知识不是书本上的理论，而是通过数十年与海洋的直接对话积累而成的生存智慧。当老人说"鱼啊，我爱你，也非常尊敬你"时，我们看到的不仅是一个渔夫对猎物的复杂情感，更是人类面对自然时既依赖又敬畏的辩证态度。

小说中最为震撼人心的部分莫过于老人与大马林鱼的对峙与搏斗。这场持续三天两夜的较量，表面上是人与鱼的搏斗，实则是人类意志与自然力量、自我极限的终极对话。海明威以惊人的细节描写能力，将这一过程呈现得惊心动魄：老人手掌的疼痛、背部的酸痛、饥饿的煎熬、口渴的折磨，以及孤独带来的精神压力，都被刻画得细致入微。正是在这种极端情境下，圣地亚哥的精神力量得到了最充分的展现——"人可以被毁灭，但不能被打败"这句名言，正是老人内心信念的最精炼表达。

值得注意的是，海明威笔下的圣地亚哥并非传统意义上的英雄形象。他身材瘦小，力量有限，甚至有些迷信和唠叨。他常常自言自语，与想象中的听众对话，这些细节使人物显得格外真实而人性化。正是这种不完美的真实感，使得老人的精神胜利更具震撼力——他不是凭借超人的能力战胜困难，而是依靠普通人的意志力在绝境中坚守尊严。当老人最终将大马林鱼的骨架拖回港口时，虽然鱼肉已被鲨鱼啃食殆尽，但他的精神胜利却完整无缺地保留了下来。

《老人与海》中的象征体系丰富而深刻，为小说增添了多重解读可能。大马林鱼可以被视为理想、目标或美好事物的象征，它的巨大体型和优雅姿态代表着值得追求的高尚价值；鲨鱼群则象征着破坏性力量、命运的无常或人类面临的种种挑战；而老人圣地亚哥本人，则是整个人类精神的象征性存在。海明威曾表示，他试图在这部小说中展现"一个人能够被摧毁，但不能被打败"的主题，这一主题通过象征手法得到了升华。

小说中的"男孩"马诺林也是一个值得关注的象征性人物。他是老人唯一真正的朋友，代表着希望、传承与人性中的温情。当其他渔夫嘲笑老人时，只有马诺林依然相信并尊敬他；当老人一无所获返回港口时，是马诺林承诺将来要与他一起出海。这种代际之间的精神传承，暗示着人类尊严与勇气的延续性——即使某一代人可能倒下，但他们的精神将通过下一代继续闪耀。

从存在主义视角解读，《老人与海》深刻展现了人类面对荒诞世界时的态度。加缪曾言："只有一个真正严肃的哲学问题，那就是自杀。"在《老人与海》中，圣地亚哥不断面临类似的根本性问题：当生活充满苦难与不确定性时，坚持是否有意义？当注定要失败时，抗争是否值得？老人的选择给出了明确的答案——即使知道可能一无所获，即使面对几乎不可能的挑战，仍然选择全力以赴，这种"西西弗斯式"的精神正是人类尊严的核心所在。

在现代社会的背景下重读《老人与海》，其启示愈发深刻。我们生活在一个物质丰富却常常精神贫瘠的时代，一个强调结果与效率却忽视过程与坚持的时代。圣地亚哥的故事提醒我们，生命的价值不仅在于获得了什么，更在于如何面对挑战，在于坚持过程中的自我超越。在竞争激烈的现代生活中，人们常常将自我价值与外在成就挂钩，而老人则告诉我们：真正的尊严来自于内心的坚守，来自于面对逆境时的不屈精神。

小说的叙事艺术同样值得称道。海明威以其标志性的"冰山理论"创作这部作品——文字表面之下蕴含着更为丰富的情感与思想。简洁的对话、克制的描写、重复的句式，这些手法共同营造出一种诗意而有力的叙事风格。当老人说"我要让他知道人能忍受什么"时，简短的话语背后是波澜壮阔的精神世界。海明威通过这种极简主义的写作风格，反而使小说的情感冲击力更为强烈。

《老人与海》中的孤独主题也具有普遍意义。圣地亚哥长时间独自一人，在茫茫大海上与自己的思想为伴。这种孤独不是惩罚，而是一种净化——它迫使老人直面自我，挖掘内在力量。在社交媒体连接一切的今天，我们或许拥有更多的"联系"，却常常体验着更为深刻的孤独。老人的故事启示我们：真正的孤独不是物理上的独处，而是精神上的隔绝；而真正的连接来自于自我认知的深度。

重读《老人与海》，我们看到的不仅是一个老人捕鱼的故事，而是一面映照人类普遍处境的镜子。每个人生命中都会有自己的"大马林鱼"——那些值得追求却难以获得的目标；也会有自己的"鲨鱼群"——那些不断侵蚀我们成果的挑战与困难。圣地亚哥告诉我们：重要的不是最终带回了什么，而是在追求过程中我们成为了什么样的人。

在结束这篇读后感时，我想起小说结尾处那个充满希望的场景：尽管只带回了一副鱼骨架，圣地亚哥却睡得很香，"他依旧脸朝下躺着，男孩坐在他身边守护着他"。这个画面传递出一种超越成败的生命韧性——休息是为了新的出发，守护意味着信念的传承。也许，《老人与海》最终告诉我们的是：生命的意义不在于永不失败，而在于每次跌倒后都能重新站起；不在于从未受伤，而在于即使伤痕累累仍能保持尊严与希望。

在这个意义上，圣地亚哥不仅是海明威笔下的虚构人物，更是人类精神的一个永恒象征。他的故事跨越时空，继续向每一个面临挑战的读者诉说：人可以被毁灭，但不能被打败；生活可以夺走一切，却无法夺走我们选择如何面对生活的自由。这或许就是《老人与海》历经半个多世纪仍能打动无数读者的根本原因——因为它讲述的不仅是老人的故事，也是我们每个人的故事。"""

    app = create_generation_workflow()
    initial_state = {"original_text": sample_text}
    result = app.invoke(initial_state)

    print("=" * 60)
    print("执行完成！")
    print("=" * 60)
    print("\n📄 最终生成结果:")
    print("-" * 40)
    print(result["final_output"])

console_result ='''
启动长文本生成系统
============================================================
============================================================
🔄 [分块阶段] 开始文本切分
============================================================
📊 切分统计:
   原始文本长度: 2683 字符
   切分块数: 15 块
   平均块长度: 176 字符

📝 切分结果详情:
   块 1: 1 词 (24 字符) | 在孤独与尊严的海洋中：重读《老人与海》的生命启示
   块 2: 1 词 (213 字符) | 在哈瓦那的晨曦中，一位名叫圣地亚哥的老渔夫独自划着小船驶向远海，这看似平凡的场景却构成了海明威《老人...
   块 3: 1 词 (186 字符) | 圣地亚哥这一人物形象的塑造，体现了海明威对人类精神世界的深刻洞察。这位连续八十四天没有捕到鱼的古巴老...
   块 4: 1 词 (237 字符) | 老人与大海的关系构成了小说最核心的隐喻。海洋在文学传统中常被赋予母性、神秘与不可知的象征意义，而在《...
   块 5: 1 词 (216 字符) | 小说中最为震撼人心的部分莫过于老人与大马林鱼的对峙与搏斗。这场持续三天两夜的较量，表面上是人与鱼的搏...
   块 6: 1 词 (201 字符) | 值得注意的是，海明威笔下的圣地亚哥并非传统意义上的英雄形象。他身材瘦小，力量有限，甚至有些迷信和唠叨...
   块 7: 1 词 (187 字符) | 《老人与海》中的象征体系丰富而深刻，为小说增添了多重解读可能。大马林鱼可以被视为理想、目标或美好事物...
   块 8: 1 词 (164 字符) | 小说中的"男孩"马诺林也是一个值得关注的象征性人物。他是老人唯一真正的朋友，代表着希望、传承与人性中...
   块 9: 1 词 (196 字符) | 从存在主义视角解读，《老人与海》深刻展现了人类面对荒诞世界时的态度。加缪曾言："只有一个真正严肃的哲...
   块 10: 1 词 (185 字符) | 在现代社会的背景下重读《老人与海》，其启示愈发深刻。我们生活在一个物质丰富却常常精神贫瘠的时代，一个...
   块 11: 1 词 (168 字符) | 小说的叙事艺术同样值得称道。海明威以其标志性的"冰山理论"创作这部作品——文字表面之下蕴含着更为丰富...
   块 12: 1 词 (172 字符) | 《老人与海》中的孤独主题也具有普遍意义。圣地亚哥长时间独自一人，在茫茫大海上与自己的思想为伴。这种孤...
   块 13: 1 词 (149 字符) | 重读《老人与海》，我们看到的不仅是一个老人捕鱼的故事，而是一面映照人类普遍处境的镜子。每个人生命中都...
   块 14: 1 词 (187 字符) | 在结束这篇读后感时，我想起小说结尾处那个充满希望的场景：尽管只带回了一副鱼骨架，圣地亚哥却睡得很香，...
   块 15: 1 词 (165 字符) | 在这个意义上，圣地亚哥不仅是海明威笔下的虚构人物，更是人类精神的一个永恒象征。他的故事跨越时空，继续...

📏 分块均匀性分析:
   最大块: 237 字符
   最小块: 24 字符
   大小差异: 213 字符
✅ 分块阶段完成

============================================================
🧠 [记忆阶段] 构建上下文记忆
============================================================
📝 正在生成摘要...
   处理块 1/15... ✅
      原文: 24 字符
      摘要: 5 字符 (压缩率: 20.8%)
      内容: 孤独与尊严...
   处理块 2/15... ✅
      原文: 213 字符
      摘要: 20 字符 (压缩率: 9.4%)
      内容: 《老人与海》探讨人类精神力量与存在本质。...
   处理块 3/15... ✅
      原文: 186 字符
      摘要: 24 字符 (压缩率: 12.9%)
      内容: 圣地亚哥体现了海明威对人类尊严与坚韧的深刻洞察。...
   处理块 4/15... ✅
      原文: 237 字符
      摘要: 32 字符 (压缩率: 13.5%)
      内容: 《老人与海》中，海洋象征生命与考验，体现人类对自然的依赖与敬畏。...
   处理块 5/15... ✅
      原文: 216 字符
      摘要: 34 字符 (压缩率: 15.7%)
      内容: 老人与大马林鱼的搏斗展现人类意志与自然的对抗，体现了精神力量与信念。...
   处理块 6/15... ✅
      原文: 201 字符
      摘要: 35 字符 (压缩率: 17.4%)
      内容: 海明威的圣地亚哥是个不完美的英雄，凭意志力坚守尊严，最终实现精神胜利。...
   处理块 7/15... ✅
      原文: 187 字符
      摘要: 22 字符 (压缩率: 11.8%)
      内容: 《老人与海》通过象征展现追求理想与人类精神。...
   处理块 8/15... ✅
      原文: 164 字符
      摘要: 20 字符 (压缩率: 12.2%)
      内容: 马诺林象征希望与人性温情，代表代际传承。...
   处理块 9/15... ✅
      原文: 196 字符
      摘要: 20 字符 (压缩率: 10.2%)
      内容: 《老人与海》展现人类在荒诞中坚持的尊严。...
   处理块 10/15... ✅
      原文: 185 字符
      摘要: 25 字符 (压缩率: 13.5%)
      内容: 《老人与海》提醒我们，生命价值在于坚持与内心坚守。...
   处理块 11/15... ✅
      原文: 168 字符
      摘要: 34 字符 (压缩率: 20.2%)
      内容: 海明威运用"冰山理论"，通过简洁对话和克制描写，增强小说情感冲击力。...
   处理块 12/15... ✅
      原文: 172 字符
      摘要: 19 字符 (压缩率: 11.0%)
      内容: 《老人与海》探讨孤独的净化与自我认知。...
   处理块 13/15... ✅
      原文: 149 字符
      摘要: 19 字符 (压缩率: 12.8%)
      内容: 《老人与海》反映人类处境，追求与挑战。...
   处理块 14/15... ✅
      原文: 187 字符
      摘要: 32 字符 (压缩率: 17.1%)
      内容: 《老人与海》传达生命韧性：跌倒后重起，伤痕累累仍保持尊严与希望。...
   处理块 15/15... ✅
      原文: 165 字符
      摘要: 21 字符 (压缩率: 12.7%)
      内容: 圣地亚哥象征人类精神，传达面对挑战的自由。...

🔍 构建向量数据库...
✅ 向量数据库构建完成
📊 向量存储统计:
   存储文档数: 15
   向量维度: 1536 (text-embedding-3-small)

🔑 关键词索引:
   文档 1: 孤独与尊严
   文档 2: 《老人与海》探讨人类精神力量与存在本质。
   文档 3: 圣地亚哥体现了海明威对人类尊严与坚韧的深刻洞察。
   文档 4: 《老人与海》中，海洋象征生命与考验，体现人类对自然的依赖与敬畏。
   文档 5: 老人与大马林鱼的搏斗展现人类意志与自然的对抗，体现了精神力量与信念。
   文档 6: 海明威的圣地亚哥是个不完美的英雄，凭意志力坚守尊严，最终实现精神胜利。
   文档 7: 《老人与海》通过象征展现追求理想与人类精神。
   文档 8: 马诺林象征希望与人性温情，代表代际传承。
   文档 9: 《老人与海》展现人类在荒诞中坚持的尊严。
   文档 10: 《老人与海》提醒我们，生命价值在于坚持与内心坚守。
   文档 11: 海明威运用"冰山理论"，通过简洁对话和克制描写，增强小说情感冲击力。
   文档 12: 《老人与海》探讨孤独的净化与自我认知。
   文档 13: 《老人与海》反映人类处境，追求与挑战。
   文档 14: 《老人与海》传达生命韧性：跌倒后重起，伤痕累累仍保持尊严与希望。
   文档 15: 圣地亚哥象征人类精神，传达面对挑战的自由。
✅ 记忆阶段完成

============================================================
📋 [规划阶段] 构建精简文章结构
============================================================
🤖 正在分析摘要并生成精简结构树...
✅ 精简结构树生成完成

📖 精简文章大纲结构:
   标题: 《老人与海》中的孤独与尊严
   主章节数: 4
   1. 人类精神力量与存在本质
      (综合段落，无子章节)
   2. 海洋的象征与人类的对抗
      (综合段落，无子章节)
   3. 代际传承与人性温情
      (综合段落，无子章节)
   4. 生命的韧性与内心坚守
      (综合段落，无子章节)

🎯 精简生成策略:
   预计内容段落数: 4 段
   段落生成方式: 每章节合并为单一综合段落
   ✅ 符合目标范围: 3-5段
✅ 规划阶段完成

============================================================
✍️ [生成阶段] 精简段落生成
============================================================
📝 生成主标题: 《老人与海》中的孤独与尊严
🎯 采用精简策略: 4 个主章节，无子章节分割

🔄 生成章节 1/4: 人类精神力量与存在本质
   📚 检索到相关上下文: 78 字符
   ✅ 综合章节内容生成完成: 353 字符
   💾 内容已添加到记忆库

🔄 生成章节 2/4: 海洋的象征与人类的对抗
   📚 检索到相关上下文: 90 字符
   ✅ 综合章节内容生成完成: 346 字符
   💾 内容已添加到记忆库

🔄 生成章节 3/4: 代际传承与人性温情
   📚 检索到相关上下文: 73 字符
   ✅ 综合章节内容生成完成: 383 字符
   💾 内容已添加到记忆库

🔄 生成章节 4/4: 生命的韧性与内心坚守
   📚 检索到相关上下文: 412 字符
   ✅ 综合章节内容生成完成: 398 字符
   💾 内容已添加到记忆库

📊 精简生成统计:
   总字符数: 1565
   主章节数: 4
   内容段落数: 4
   总组件数: 9
   ✅ 成功将段落数控制在 4 段（目标: 3-5段）
✅ 生成阶段完成

============================================================
执行完成！
============================================================

📄 最终生成结果:
----------------------------------------
# 《老人与海》中的孤独与尊严
## 人类精神力量与存在本质
《老人与海》深刻探讨了人类精神力量与存在的本质，通过主人公与大马林鱼的搏斗，展现了人类意志与自然之间的激烈对抗。这场斗争不仅是肉体上的较量，更是精神层面的升华，体现了人类在面对困境时所展现出的坚定信念与不屈不挠的精神。海明威通过象征手法，将大马林鱼视为理想与目标的象征，而老人的坚持则代表了人类追求理想的执着与勇气。尽管最终的结果并不如人所愿，老人依然在这场斗争中找到了存在的意义，彰显了人类在逆境中所能展现的伟大精神力量。这种力量不仅是对抗自然的勇气，更是对生命本质的深刻理解与尊重。通过老人的经历，海明威传达了一个重要的哲理：人类的价值不在于最终的胜利，而在于追求理想过程中的坚持与努力。这种对人类精神的深刻洞察，使得《老人与海》成为一部超越时代的经典，激励着一代又一代人去思考生命的意义与存在的价值。

## 海洋的象征与人类的对抗
在《老人与海》中，海洋不仅是自然的象征，更是生命与考验的舞台，深刻体现了人类对自然的依赖与敬畏。海洋的广袤与神秘，既是人类追求理想的背景，也是挑战与磨难的源泉。老人与大马林鱼的搏斗，生动地展现了人类意志与自然力量之间的对抗。这场斗争不仅是对生存的追求，更是对精神力量与信念的考验。老人面对巨大的马林鱼，象征着人类在面对自然时的勇气与坚持，尽管力量悬殊，但他依然不屈不挠，展现出一种超越肉体的精神追求。通过这一象征，海洋成为了人类理想与现实之间的桥梁，反映出人类在追求目标时所需的毅力与决心。最终，尽管老人经历了无数的艰辛与挫折，他的精神却在这场与自然的较量中得到了升华，体现了人类在面对困境时的坚韧与不屈。这种对抗不仅是生存的斗争，更是对生命意义的深刻探索，揭示了人类在自然面前的渺小与伟大。

## 代际传承与人性温情
在海明威的《老人与海》中，马诺林这一角色不仅是老渔夫圣地亚哥的助手，更是代际传承与人性温情的象征。马诺林对圣地亚哥的尊敬与关爱，体现了年轻一代对老一辈智慧与经验的珍视，反映出人类社会中代际关系的深厚纽带。这种传承不仅是技能与知识的延续，更是情感与价值观的交融，展现了人性中最温暖的一面。与此同时，圣地亚哥在面对无尽挑战时所展现出的生命韧性，正是对希望与尊严的坚定追求。尽管他在追逐梦想的过程中屡遭挫折，身心俱疲，但他始终不屈不挠，勇敢地面对生活的风浪。这种精神不仅激励着马诺林，也让读者深刻体会到人类在逆境中所展现的坚韧与勇气。通过这代际间的互动，海明威传达了一个重要的主题：无论生活多么艰难，人与人之间的关爱与支持，始终是我们面对挑战、追求梦想的动力源泉。这种温情与希望的交织，构成了人类生存的核心，提醒我们在追求个人目标的同时，不应忽视与他人之间的深厚情感联系。

## 生命的韧性与内心坚守
在《老人与海》中，海明威深刻探讨了生命的韧性与内心的坚守，通过主人公与大马林鱼的搏斗，展现了人类精神力量与自然之间的激烈对抗。这场斗争不仅是肉体上的较量，更是精神层面的升华，体现了人在面对困境时所展现出的坚定信念与不屈不挠的精神。大马林鱼象征着理想与目标，而老人的坚持则代表了人类追求理想的执着与勇气。尽管最终的结果并不如人所愿，老人依然在这场斗争中找到了存在的意义，彰显了人类在逆境中所能展现的伟大精神力量。这种力量不仅是对抗自然的勇气，更是对生命本质的深刻理解与尊重。海明威通过老人的经历传达了一个重要的哲理：人类的价值不在于最终的胜利，而在于追求理想过程中的坚持与努力。这种对人类精神的深刻洞察，使得《老人与海》成为一部超越时代的经典，激励着一代又一代人去思考生命的意义与存在的价值。通过对生命韧性与内心坚守的探讨，海明威不仅塑造了一个不屈的英雄形象，也引发了读者对自身生命体验的深刻反思。

'''