"""
09 - 进阶检索：五种召回策略同题对比

运行:  python src/09_advanced_retrieval.py [问题]

这个脚本用同一个问题跑五种检索配置，把结果并排打出来，
让你直接看到差异，而不是听结论：

  vector          单路向量。语义近似强，精确串弱
  mmr             向量 + 多样性惩罚。抑制近重复片段占满 top-k
  bm25            纯字面命中。专有名词/错误码/函数名的兜底
  hybrid(rrf)     向量 + BM25，按名次融合。生产默认
  hybrid+rerank   融合后再用 LLM 精排（慢，但精度最高）

建议试三类问题，你会看到三种截然不同的排名格局：
  A. 概念题「RAG 为什么能减少幻觉」        → 向量占优
  B. 精确串「check_embedding_ctx_length 是什么」→ BM25 占优
  C. 混合题「chunk_size 设多大合适」        → 混合明显最稳
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_llm  # noqa: E402
from rag import HybridRetriever, ensure_kb  # noqa: E402

DEFAULT_QUESTION = "chunk_size 设多大合适，overlap 怎么取？"

# (标签, 构造参数)
STRATEGIES = [
    ("vector", dict(mode="vector")),
    ("mmr", dict(mode="mmr")),
    ("bm25", dict(mode="bm25")),
    ("hybrid(rrf)", dict(mode="hybrid", fusion="rrf")),
    ("hybrid(weighted)", dict(mode="hybrid", fusion="weighted")),
    ("hybrid+rerank", dict(mode="hybrid", fusion="rrf", use_rerank=True)),
]


def main():
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    kb = ensure_kb()
    llm = get_llm(temperature=0)

    print("=" * 78)
    print(f"问题: {question}")
    print("=" * 78)

    for label, kwargs in STRATEGIES:
        retriever = HybridRetriever(kb=kb, top_k=3, fetch_k=12, llm=llm, **kwargs)
        result = retriever.search(question)
        print(f"\n▶ {label}  候选 {result.candidates} 条 → 取 3 条  耗时 {result.timings_ms}")
        print(result.table())

    print("\n" + "=" * 78)
    print("观察要点：")
    print("  1. 对比 vector 和 bm25 的命中片段，看两者互补在哪里")
    print("  2. hybrid 的 top1 通常来自「两个通道都召回」的那条 —— 这是 RRF 的作用")
    print("  3. rerank 会改变排序但不增加候选，所以它救不了「召回阶段就没捞到」的情况")
    print("  4. 看耗时：rerank 往往比检索本身贵一个数量级")
    print("=" * 78)


if __name__ == "__main__":
    main()
