"""
RAG 离线评估。

这是原项目完全缺失、但对 RAG 最关键的一环。没有评估，
「换个 chunk_size」「加个重排」的收益全靠感觉，
而 RAG 的调参空间大到根本不可能靠感觉找到好配置。

评估分两层，必须分开看：

【检索层】决定天花板。答案里不可能出现没被检索到的信息。
  Hit@k    top-k 中至少命中一个标准来源的比例。最直观的健康度指标
  Recall@k 标准来源被召回的比例（一个问题对应多个来源时才有区分度）
  MRR      首个正确结果的名次倒数均值。衡量「正确答案排得够不够前」
  nDCG@k   带位置折扣的增益。综合考虑命中数量与名次

【生成层】决定实际体验。
  忠实度 Faithfulness  答案是否只用了检索到的内容（幻觉检测）—— LLM 评判
  相关度 Relevance     答案是否真正回答了问题 —— LLM 评判
  关键词覆盖            标准答案关键词的命中率 —— 无需 LLM，快且稳定，适合回归测试
  拒答率                检索不到时是否老实拒答

排查顺序永远是：检索层指标差 → 先修索引和召回，别碰提示词。
检索层好但生成层差 → 才是提示词/模型的问题。

数据集格式（JSON 数组）：
  [{"question": "...",
    "expected_sources": ["rag_basics.md"],   # 文件名，子串匹配
    "expected_keywords": ["BM25", "倒排"],    # 标准答案应包含的关键词
    "reference_answer": "..."                 # 可选，供人工查阅
   }, ...]
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .pipeline import ABSTAIN_MESSAGE, RagConfig, RagPipeline

# ===== 检索指标 =====


def _matched(doc_meta: dict, expected_sources: List[str]) -> bool:
    """来源是否命中。用子串匹配，避免绝对路径与文件名写法不一致导致全部误判。"""
    actual = f"{doc_meta.get('source', '')}|{doc_meta.get('file_name', '')}".lower()
    return any(exp.lower() in actual for exp in expected_sources)


def hit_at_k(hits: List[bool]) -> float:
    return 1.0 if any(hits) else 0.0


def recall_at_k(matched_sources: set, expected_sources: List[str]) -> float:
    if not expected_sources:
        return 0.0
    return len(matched_sources) / len(set(s.lower() for s in expected_sources))


def mrr(hits: List[bool]) -> float:
    for i, h in enumerate(hits, 1):
        if h:
            return 1.0 / i
    return 0.0


def ndcg_at_k(hits: List[bool], ideal_count: int) -> float:
    """二元相关性下的 nDCG：增益为 1/0，折扣为 1/log2(rank+1)。

    理想排序里的相关文档数必须不小于实际命中数，否则 IDCG 偏小、nDCG 会超过 1。
    一个来源常常对应多个命中 chunk，所以不能直接拿 expected_sources 的个数当上界。
    """
    relevant = sum(1 for h in hits if h)
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, h in enumerate(hits, 1) if h)
    ideal_n = max(relevant, min(ideal_count, len(hits)))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


# ===== 生成指标（LLM-as-judge）=====

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template(
    "你是 RAG 答案审核员。判断【答案】中的事实陈述是否都能从【资料】中找到依据。\n"
    "评分：1 = 大量内容无依据（严重幻觉）；3 = 主要内容有依据但有少量外推；"
    "5 = 全部有依据。\n只输出一个 1-5 的整数。\n\n"
    "资料:\n{context}\n\n答案:\n{answer}\n\n分数:"
)

_RELEVANCE_PROMPT = ChatPromptTemplate.from_template(
    "你是问答质量审核员。判断【答案】是否切题地回答了【问题】（不考虑事实对错）。\n"
    "评分：1 = 完全没回答；3 = 部分回答或答偏；5 = 完整切题。\n"
    "只输出一个 1-5 的整数。\n\n问题: {question}\n\n答案:\n{answer}\n\n分数:"
)


def _judge(chain, payload: dict) -> Optional[float]:
    """跑一次 LLM 评判。解析失败返回 None，由上层从均值里排除。"""
    try:
        raw = chain.invoke(payload)
    except Exception as e:
        print(f"  [warn] LLM 评判失败: {e}")
        return None
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    return max(1.0, min(5.0, float(digits[0])))


def keyword_coverage(answer: str, keywords: List[str]) -> Optional[float]:
    if not keywords:
        return None
    low = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in low) / len(keywords)


# ===== 数据集 =====


@dataclass
class EvalCase:
    question: str
    expected_sources: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    reference_answer: str = ""


def load_dataset(path: str) -> List[EvalCase]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        EvalCase(
            question=item["question"],
            expected_sources=item.get("expected_sources", []),
            expected_keywords=item.get("expected_keywords", []),
            reference_answer=item.get("reference_answer", ""),
        )
        for item in raw
    ]


# ===== 主评估流程 =====


@dataclass
class CaseResult:
    question: str
    hit: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    faithfulness: Optional[float] = None
    relevance: Optional[float] = None
    keyword_coverage: Optional[float] = None
    abstained: bool = False
    latency_ms: float = 0.0
    answer: str = ""
    retrieved_sources: List[str] = field(default_factory=list)


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def evaluate_retrieval(pipeline: RagPipeline, cases: List[EvalCase]) -> List[CaseResult]:
    """只评检索层。不调用生成，跑得很快，适合密集调参。"""
    results = []
    for case in cases:
        t = time.perf_counter()
        retrieval = pipeline.retriever.search(case.question)
        latency = (time.perf_counter() - t) * 1000

        hits = [_matched(d.metadata or {}, case.expected_sources) for d in retrieval.documents]
        matched_sources = {
            exp.lower()
            for exp in case.expected_sources
            if any(_matched(d.metadata or {}, [exp]) for d in retrieval.documents)
        }
        results.append(
            CaseResult(
                question=case.question,
                hit=hit_at_k(hits),
                recall=recall_at_k(matched_sources, case.expected_sources),
                mrr=mrr(hits),
                ndcg=ndcg_at_k(hits, len(case.expected_sources) or 1),
                latency_ms=round(latency, 1),
                retrieved_sources=[
                    str((d.metadata or {}).get("file_name") or (d.metadata or {}).get("source", "?"))
                    for d in retrieval.documents
                ],
            )
        )
    return results


def evaluate(
    pipeline: RagPipeline,
    cases: List[EvalCase],
    judge_llm=None,
    verbose: bool = True,
) -> dict:
    """端到端评估。judge_llm 为 None 时跳过 LLM 评判，只算可确定性计算的指标。

    生产实践：judge 最好用比被测模型更强的模型，否则「自己给自己打分」
    会系统性偏高。本地只有一个模型时，重点看关键词覆盖和检索层指标。
    """
    faith_chain = (_FAITHFULNESS_PROMPT | judge_llm | StrOutputParser()) if judge_llm else None
    rel_chain = (_RELEVANCE_PROMPT | judge_llm | StrOutputParser()) if judge_llm else None

    results: List[CaseResult] = []
    for idx, case in enumerate(cases, 1):
        if verbose:
            print(f"  [{idx}/{len(cases)}] {case.question}")

        t = time.perf_counter()
        ans = pipeline.answer(case.question)
        latency = (time.perf_counter() - t) * 1000

        hits = [_matched(d.metadata or {}, case.expected_sources) for d in ans.documents]
        matched_sources = {
            exp.lower()
            for exp in case.expected_sources
            if any(_matched(d.metadata or {}, [exp]) for d in ans.documents)
        }

        res = CaseResult(
            question=case.question,
            hit=hit_at_k(hits),
            recall=recall_at_k(matched_sources, case.expected_sources),
            mrr=mrr(hits),
            ndcg=ndcg_at_k(hits, len(case.expected_sources) or 1),
            keyword_coverage=keyword_coverage(ans.answer, case.expected_keywords),
            abstained=ans.abstained or ans.answer.strip() == ABSTAIN_MESSAGE,
            latency_ms=round(latency, 1),
            answer=ans.answer,
            retrieved_sources=[
                str((d.metadata or {}).get("file_name") or (d.metadata or {}).get("source", "?"))
                for d in ans.documents
            ],
        )

        # 拒答的样本不参与生成质量打分，否则会把指标算歪
        if faith_chain and not res.abstained:
            context = "\n\n".join(d.page_content for d in ans.documents)
            res.faithfulness = _judge(faith_chain, {"context": context, "answer": ans.answer})
            res.relevance = _judge(rel_chain, {"question": case.question, "answer": ans.answer})

        results.append(res)

    n = len(results) or 1
    summary = {
        "config": pipeline.config.label(),
        "cases": len(results),
        "hit@k": round(sum(r.hit for r in results) / n, 3),
        "recall@k": round(sum(r.recall for r in results) / n, 3),
        "mrr": round(sum(r.mrr for r in results) / n, 3),
        "ndcg@k": round(sum(r.ndcg for r in results) / n, 3),
        "keyword_coverage": _mean([r.keyword_coverage for r in results]),
        "faithfulness": _mean([r.faithfulness for r in results]),
        "relevance": _mean([r.relevance for r in results]),
        "abstain_rate": round(sum(1 for r in results if r.abstained) / n, 3),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 1),
    }
    return {"summary": summary, "cases": results}


def compare_configs(
    configs: List[RagConfig],
    cases: List[EvalCase],
    judge_llm=None,
    retrieval_only: bool = False,
) -> List[dict]:
    """跑多组配置做 A/B，返回各自的汇总指标。

    retrieval_only=True 时不调生成，速度快一个数量级，
    调索引和召回参数时应该用这个模式。
    """
    pipeline: Optional[RagPipeline] = None
    summaries = []
    for cfg in configs:
        print(f"\n>>> 配置: {cfg.label()}")
        pipeline = RagPipeline(cfg) if pipeline is None else pipeline.reconfigure(cfg)

        if retrieval_only:
            results = evaluate_retrieval(pipeline, cases)
            n = len(results) or 1
            summaries.append(
                {
                    "config": cfg.label(),
                    "cases": len(results),
                    "hit@k": round(sum(r.hit for r in results) / n, 3),
                    "recall@k": round(sum(r.recall for r in results) / n, 3),
                    "mrr": round(sum(r.mrr for r in results) / n, 3),
                    "ndcg@k": round(sum(r.ndcg for r in results) / n, 3),
                    "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 1),
                }
            )
        else:
            summaries.append(evaluate(pipeline, cases, judge_llm)["summary"])
    return summaries


def render_table(rows: List[dict]) -> str:
    """把汇总指标渲染成等宽表格（不依赖 pandas，容器里也能跑）。"""
    if not rows:
        return "(无数据)"
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    def cell(v) -> str:
        return "-" if v is None else (f"{v}" if not isinstance(v, float) else f"{v:g}")

    widths = {
        c: max(len(c), *(len(cell(r.get(c))) for r in rows)) for c in columns
    }
    sep = "-+-".join("-" * widths[c] for c in columns)
    head = " | ".join(c.ljust(widths[c]) for c in columns)
    body = [
        " | ".join(cell(r.get(c)).ljust(widths[c]) for c in columns) for r in rows
    ]
    return "\n".join([head, sep, *body])


def save_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "summary": report["summary"],
        "cases": [
            {k: v for k, v in vars(c).items()} for c in report.get("cases", [])
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
