"""
07 - LangChain 链式编排: Map-Reduce 长文档摘要

运行:  python 07_map_reduce_summarize.py

场景: 对超长文档生成摘要。单次把整篇文档丢给模型会超出上下文窗口，
      Map-Reduce 策略解决了这个问题:

      1. Map 阶段: 把长文档切成多个块，对每块分别生成摘要
      2. Reduce 阶段: 把所有块摘要合并，生成最终全局摘要

工作流:
    长文档
      │
      ▼
    切分为 N 个文本块
      │
      ▼
    ┌────────┬────────┬────────┐
    │ Map 1  │ Map 2  │ Map N  │   ← 每块独立生成摘要（可并行）
    │ 摘要1  │ 摘要2  │ 摘要N  │
    └────────┴────────┴────────┘
                │
                ▼
          合并所有摘要
                │
                ▼
          Reduce: 生成最终摘要

对比说明:
    - Stuff 方式: 把所有内容塞进一个 prompt（适合短文档）
    - Map-Reduce 方式: 分块处理再合并（适合超长文档）

注: LangChain 1.0+ 移除了 load_summarize_chain，这里用 LCEL (LangChain Expression Language)
    手动构建链，逻辑更透明，也便于理解 Map-Reduce 的本质。
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_llm

# 一篇用于演示的"长文档"（关于 AI 发展史）
LONG_DOCUMENT = """
人工智能（AI）的概念最早可以追溯到1950年代。1950年，艾伦·图灵提出了著名的"图灵测试"，
认为如果一台机器能让人类无法分辨它是机器还是人，就可以认为它具有智能。
1956年的达特茅斯会议上，约翰·麦卡锡首次提出"人工智能"一词，标志着AI作为独立学科的诞生。

1960-70年代，AI研究集中于符号推理和专家系统。研究人员相信，通过编码人类知识规则，
机器就能实现智能。这一时期涌现了如DENDRAL（化学分析专家系统）和MYCIN（医学诊断系统）等成果。
然而，专家系统需要大量人工编码规则，难以应对现实世界的复杂性，AI进入了第一次"寒冬"。

1980年代，机器学习开始兴起。与符号推理不同，机器学习让计算机从数据中自动学习规律。
决策树、支持向量机等算法被广泛应用。这一时期，神经网络研究也取得突破，
反向传播算法的普及使多层神经网络的训练成为可能。

1997年，IBM的深蓝击败国际象棋世界冠军卡斯帕罗夫，引发轰动。
这标志着在特定领域，机器的计算能力已超越人类顶尖选手。
然而，深蓝主要依靠暴力搜索而非真正的"智能"。

2012年是深度学习的转折年。AlexNet在ImageNet图像识别竞赛中以巨大优势夺冠，
证明了深度卷积神经网络的力量。此后，深度学习在语音识别、图像分类、
自然语言处理等领域接连突破。GPU的普及和大数据的积累为深度学习提供了硬件和数据基础。

2017年，Google发表论文《Attention Is All You Need》，提出Transformer架构。
Transformer摒弃了循环结构，完全基于注意力机制，能更高效地处理序列数据。
这一架构成为后来所有大语言模型的基础。

2020年，OpenAI发布GPT-3，拥有1750亿参数，展现出惊人的语言理解和生成能力。
GPT-3能写文章、写代码、做翻译，无需专门微调就能完成各种任务（few-shot learning）。
这标志着大模型时代的到来。

2022年底，ChatGPT发布，引发全球AI热潮。ChatGPT基于GPT-3.5，通过人类反馈强化学习（RLHF）
进行对齐训练，使回答更符合人类期望。ChatGPT上线两个月用户破亿，成为史上增长最快的应用。

2023-2024年，GPT-4、Claude、Gemini、Qwen等大模型百花齐放。
多模态能力（理解图像、视频）、超长上下文（百万token）、工具调用、Agent等能力不断突破。
AI开始从"对话工具"向"智能助手"演进。

与此同时，AI也面临挑战：幻觉问题（生成看似合理但错误的内容）、偏见与安全性、
高昂的训练和推理成本、以及版权和伦理争议。RAG（检索增强生成）、
RLHF对齐、模型压缩等技术被用于缓解这些问题。

展望未来，AI正朝着通用人工智能（AGI）的方向发展。多智能体协作、自主规划与执行、
持续学习等方向是当前研究热点。AI有望在科学研究、医疗、教育等领域发挥更大价值，
但也需要社会各界共同应对其带来的挑战。
"""


def main():
    llm = get_llm(temperature=0.3)

    # 将长文档转为 Document 对象
    doc = Document(page_content=LONG_DOCUMENT.strip())
    print(f"原文长度: {len(doc.page_content)} 字符\n")

    # ===== 方式一: Stuff（直接塞入，适合短文档） =====
    # 用 LCEL 构建: prompt | llm | parser
    print("=" * 60)
    print("  方式一: Stuff（整体摘要）")
    print("=" * 60)

    stuff_prompt = ChatPromptTemplate.from_template(
        "请用中文写一段简洁的摘要，概括以下内容的核心要点:\n\n{text}\n\n摘要:"
    )
    stuff_chain = stuff_prompt | llm | StrOutputParser()
    stuff_summary = stuff_chain.invoke({"text": doc.page_content})
    print(stuff_summary)

    # ===== 方式二: Map-Reduce（分块摘要再合并，适合长文档） =====
    print("\n" + "=" * 60)
    print("  方式二: Map-Reduce（分块摘要 → 合并）")
    print("=" * 60)

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    print(f"切分为 {len(chunks)} 个文本块\n")

    # --- Map 阶段: 对每个块独立生成摘要 ---
    map_prompt = ChatPromptTemplate.from_template(
        "请用中文写一段简洁的摘要，概括以下内容:\n\n{text}\n\n摘要:"
    )
    map_chain = map_prompt | llm | StrOutputParser()

    print("Map 阶段: 逐块生成摘要...")
    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        summary = map_chain.invoke({"text": chunk.page_content})
        chunk_summaries.append(summary)
        print(f"  块 {i}/{len(chunks)} 完成 ({len(summary)} 字符)")

    # --- Reduce 阶段: 合并所有块摘要，生成最终摘要 ---
    print("\nReduce 阶段: 合并生成最终摘要...")
    combined = "\n\n".join(
        f"【片段{i}】{s}" for i, s in enumerate(chunk_summaries, 1)
    )
    reduce_prompt = ChatPromptTemplate.from_template(
        "以下是一篇长文档各片段的摘要。请将它们合并成一段连贯、完整的最终摘要，"
        "保留关键信息，去除重复:\n\n{summaries}\n\n最终摘要:"
    )
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    mr_summary = reduce_chain.invoke({"summaries": combined})
    print(mr_summary)

    # ===== 对比 =====
    print("\n" + "=" * 60)
    print("  对比说明")
    print("=" * 60)
    print(f"Stuff 摘要长度:       {len(stuff_summary)} 字符")
    print(f"Map-Reduce 摘要长度:  {len(mr_summary)} 字符")
    print(f"Map-Reduce LLM 调用:  {len(chunks)}(map) + 1(reduce) = {len(chunks) + 1} 次")
    print("\n对于真正的超长文档（如整本书），Stuff 会超出上下文窗口，")
    print("而 Map-Reduce 能稳定工作——代价是更多的 LLM 调用次数。")
    print("若需进一步提升速度，Map 阶段可用 asyncio 并行调用。")


if __name__ == "__main__":
    main()
