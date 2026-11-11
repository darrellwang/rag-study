"""
重排（rerank）：把召回的 20 条精挑成最相关的 4 条。

为什么需要单独一层？
召回阶段为了速度，用的是「查询向量 vs 文档向量」的独立编码（bi-encoder），
查询和文档从来没有相互看见过。重排则让两者一起进模型（cross-encoder 或 LLM），
判断精度高得多，但代价是 O(候选数) 次推理，所以只能作用在少量候选上。

标准配方：向量+BM25 召回 fetch_k=20 → 重排 → 取 top_k=4 给生成。

本模块提供两种实现：
  llm_rerank            用本地 LLM 打 0-10 分。零额外依赖，但慢（靠 batch 并发缓解）
  cross_encoder_rerank  用 sentence-transformers 的 cross-encoder。快且准，
                        需要额外装包并下载模型，作为可选路径

注意 llm_rerank 用的是 chain.batch()，它会并发发请求。LMStudio 单实例并发能力
有限，max_concurrency 设为 4 左右比较稳，设太高会排队甚至超时。
"""
from __future__ import annotations

import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

_SCORE_PROMPT = ChatPromptTemplate.from_template(
    "你是检索结果相关性评分器。请评估下面的文档片段对回答该问题的有用程度。\n"
    "评分标准：0 = 完全无关；5 = 有一定关联但不足以回答；10 = 直接包含答案。\n"
    "只输出一个 0 到 10 的整数，不要任何解释。\n\n"
    "问题: {question}\n\n文档片段:\n{document}\n\n分数:"
)

_INT = re.compile(r"\d+")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _parse_score(text: str) -> float:
    """从模型输出里抽出分数。

    两个细节：
    · 先剥掉 <think>...</think>。有些服务端把思维链混在正文里返回，
      里面的数字（"评分标准 0 到 10"）会被误当成分数。
    · 取最后一个整数而不是第一个。模型即使被要求只输出数字，
      也常写成「这段文档直接回答了问题，评分：9」，答案在末尾。
    解析彻底失败时给中性分，避免误杀候选。
    """
    cleaned = _THINK_BLOCK.sub("", text or "").strip()
    matches = _INT.findall(cleaned)
    if not matches:
        return 5.0
    return max(0.0, min(10.0, float(matches[-1])))


def llm_rerank(
    question: str,
    docs: List[Document],
    llm=None,
    top_k: int = 4,
    min_score: float = 3.0,
    max_concurrency: int = 4,
) -> List[Tuple[Document, float]]:
    """用 LLM 给候选打分并重排，返回 (文档, 分数) 降序列表。

    llm 为 None 时使用 config.get_scoring_llm()——关闭了思维链的打分专用模型。
    别把普通对话模型传进来：推理模型为了吐一个数字会先生成上千个 reasoning
    token，重排耗时会从秒级变成分钟级。

    min_score 起到「相关性门槛」的作用：分数过低的候选直接丢弃，
    宁可少给上下文也不要把噪声塞进提示词——这比截断 top_k 更能抑制幻觉。
    """
    if not docs:
        return []
    if llm is None:
        from config import get_scoring_llm

        llm = get_scoring_llm()

    chain = _SCORE_PROMPT | llm | StrOutputParser()
    inputs = [{"question": question, "document": d.page_content} for d in docs]
    try:
        outputs = chain.batch(inputs, config={"max_concurrency": max_concurrency})
    except Exception as e:
        print(f"  [warn] rerank 失败，退回原顺序: {e}")
        return [(d, 0.0) for d in docs[:top_k]]

    scored = [(d, _parse_score(o)) for d, o in zip(docs, outputs)]
    scored.sort(key=lambda x: x[1], reverse=True)

    kept = [(d, s) for d, s in scored if s >= min_score]
    # 全部低于门槛时保留最高分那条，让上层自行判断是否拒答
    if not kept and scored:
        kept = scored[:1]
    return kept[:top_k]


def cross_encoder_rerank(
    question: str,
    docs: List[Document],
    top_k: int = 4,
    model_name: str = "BAAI/bge-reranker-base",
) -> List[Tuple[Document, float]]:
    """可选路径：cross-encoder 重排。未安装依赖时抛 ImportError 由调用方兜底。"""
    from sentence_transformers import CrossEncoder  # 延迟导入，属可选依赖

    model = CrossEncoder(model_name)
    pairs = [(question, d.page_content) for d in docs]
    scores = model.predict(pairs)
    scored = sorted(zip(docs, (float(s) for s in scores)), key=lambda x: x[1], reverse=True)
    return scored[:top_k]
