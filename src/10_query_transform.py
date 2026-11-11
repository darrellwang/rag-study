"""
10 - 查询改写：Multi-Query / HyDE / 问题分解

运行:  python src/10_query_transform.py [问题]

检索失败经常不是因为知识库里没有答案，而是因为「用户的问法」和
「文档的写法」对不上。这个脚本展示三种在检索前加工问题的手段，
并对比改写前后的召回变化。

  Multi-Query  生成多个等价改写，多点覆盖语义邻域。最便宜、最有效
  HyDE         先写一段假设性答案，用答案去检索。文体更贴近文档
  问题分解      把复合问题拆成子问题分别检索

代价提醒：每种手段都要额外调用大模型。本地模型下 Multi-Query 会让
延迟翻两三倍，HyDE 更贵。所以务必用 11_rag_eval.py 验证收益再决定是否开启。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_llm  # noqa: E402
from rag import (  # noqa: E402
    HybridRetriever,
    decompose_question,
    ensure_kb,
    hyde_document,
    multi_query,
)

# 故意用「口语化、和文档措辞不一致」的问法，这才是改写发挥作用的场景
DEFAULT_QUESTION = "同一个文件传了两遍，库里内容重了怎么办？"


def show(label: str, docs) -> None:
    print(f"\n▶ {label}")
    if not docs:
        print("  (无命中)")
        return
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("file_name", "?")
        print(f"  [{i}] {src}: {d.page_content.replace(chr(10), ' ')[:80]}...")


def main():
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    kb = ensure_kb()
    llm = get_llm(temperature=0)

    print("=" * 78)
    print(f"原问题: {question}")
    print("=" * 78)

    # ---- 1. 基线：直接检索 ----
    base = HybridRetriever(kb=kb, mode="hybrid", top_k=3, llm=llm)
    show(f"基线 hybrid（耗时 {base.search(question).timings_ms.get('total')}ms）", base.invoke(question))

    # ---- 2. Multi-Query ----
    print("\n" + "-" * 78)
    variants = multi_query(question, llm, n=3)
    print("Multi-Query 生成的改写:")
    for v in variants:
        print(f"  · {v}")
    mq = HybridRetriever(kb=kb, mode="hybrid", top_k=3, llm=llm, use_multi_query=True)
    r = mq.search(question)
    show(f"Multi-Query 检索（候选 {r.candidates} 条，耗时 {r.timings_ms}）", r.documents)

    # ---- 3. HyDE ----
    print("\n" + "-" * 78)
    hypothetical = hyde_document(question, llm)
    print(f"HyDE 生成的假设性答案:\n  {hypothetical[:200]}...")
    hyde = HybridRetriever(kb=kb, mode="hybrid", top_k=3, llm=llm, use_hyde=True)
    r = hyde.search(question)
    show(f"HyDE 检索（候选 {r.candidates} 条，耗时 {r.timings_ms}）", r.documents)

    # ---- 4. 问题分解 ----
    print("\n" + "-" * 78)
    compound = "混合检索和重排分别解决什么问题，两者能互相替代吗？"
    print(f"复合问题: {compound}")
    subs = decompose_question(compound, llm, n=3)
    print("拆解出的子问题:")
    for s in subs:
        print(f"  · {s}")
    merged, seen = [], set()
    for s in subs:
        for d in base.invoke(s):
            key = d.page_content[:80]
            if key not in seen:
                seen.add(key)
                merged.append(d)
    show(f"子问题合并去重后共 {len(merged)} 条", merged[:5])

    print("\n" + "=" * 78)
    print("观察要点：")
    print("  1. 口语化问题下，Multi-Query 常能把基线漏掉的文档捞回来")
    print("  2. HyDE 生成的术语若跑偏，检索也会跟着跑偏 —— 它是双刃剑")
    print("  3. 复合问题单次检索通常只能覆盖一半，分解后覆盖面明显变宽")
    print("  4. 收益必须用评估集量化，不要凭这一个例子下结论")
    print("=" * 78)


if __name__ == "__main__":
    main()
