"""
查询改写（query transformation）——在检索前先加工问题。

用户的问题和文档的表述往往不在同一个"语言空间"：
用户问「登录一直转圈怎么办」，文档写的是「认证服务超时排查」。
单条 query 的向量只能落在一个点上，命中与否运气成分很大。三种改法：

1. Multi-Query：让模型生成 N 个不同角度的改写，各自检索后合并去重。
   本质是用多个采样点覆盖语义邻域，最便宜且最有效的召回提升手段。

2. HyDE（Hypothetical Document Embeddings）：先让模型「凭空写一段答案」，
   然后用这段假设性答案去检索。原理是答案和文档的文体、术语更接近，
   比问题本身更容易匹配上。代价是模型幻觉出的术语可能带偏检索，
   所以适合有一定领域常识的模型 + 术语密集的知识库。

3. 问题分解：把复合问题拆成子问题分别检索。
   「A 和 B 有什么区别」这类问题，单次检索通常只能召回 A 或 B 的资料。

工程提醒：这三种都要额外 LLM 调用，本地模型下延迟成本很高。
上线前务必用 evaluate 模块确认收益，别默认全开。
"""
from __future__ import annotations

from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    "你是检索查询改写助手。请针对下面的问题，生成 {n} 个语义等价但表述不同的检索查询，"
    "覆盖不同的用词和角度（例如换成同义术语、换成更具体或更概括的说法）。\n"
    "要求：每行一个查询，不要编号，不要解释，不要输出原问题以外的内容。\n\n"
    "原问题: {question}\n\n改写:"
)

_HYDE_PROMPT = ChatPromptTemplate.from_template(
    "请针对下面的问题，写一段 100 字左右、像技术文档一样的假设性答案。"
    "即使你不确定事实细节也要写，重点是使用该领域应有的专业术语和表述方式。"
    "只输出这段文字。\n\n问题: {question}\n\n文档段落:"
)

_DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    "请把下面的问题拆解成 2 到 {n} 个可独立检索的子问题。"
    "若问题本身已经足够单一，就原样输出一行。\n"
    "要求：每行一个子问题，不要编号，不要解释。\n\n原问题: {question}\n\n子问题:"
)


def _lines(text: str, limit: int) -> List[str]:
    """按行解析模型输出，顺带去掉常见的编号/项目符号前缀。"""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-*•").strip()
        # 去掉 "1. " / "1) " 这类前缀
        if line[:1].isdigit():
            for sep in (". ", ") ", "、", ". ", "："):
                if sep in line[:4]:
                    line = line.split(sep, 1)[1].strip()
                    break
        if line and line not in out:
            out.append(line)
        if len(out) >= limit:
            break
    return out


def multi_query(question: str, llm, n: int = 3) -> List[str]:
    """生成 n 个改写查询，返回值第一个始终是原问题（保底通道）。"""
    chain = _MULTI_QUERY_PROMPT | llm | StrOutputParser()
    try:
        variants = _lines(chain.invoke({"question": question, "n": n}), n)
    except Exception as e:
        print(f"  [warn] multi-query 生成失败: {e}")
        variants = []
    return [question] + [v for v in variants if v != question]


def hyde_document(question: str, llm) -> str:
    """生成假设性答案文本。失败时退回原问题，保证链路不中断。"""
    chain = _HYDE_PROMPT | llm | StrOutputParser()
    try:
        text = chain.invoke({"question": question}).strip()
        return text or question
    except Exception as e:
        print(f"  [warn] HyDE 生成失败: {e}")
        return question


def decompose_question(question: str, llm, n: int = 3) -> List[str]:
    """拆解复合问题为子问题列表。"""
    chain = _DECOMPOSE_PROMPT | llm | StrOutputParser()
    try:
        subs = _lines(chain.invoke({"question": question, "n": n}), n)
    except Exception as e:
        print(f"  [warn] 问题分解失败: {e}")
        subs = []
    return subs or [question]
