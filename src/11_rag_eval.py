"""
11 - RAG 评估：把「感觉变好了」变成可比较的数字

运行:
  # 1) 纯检索 A/B（只调 embedding，16 条样本几秒跑完）—— 一定先跑这个
  python src/11_rag_eval.py --retrieval-only --all

  # 2) 端到端评估单个配置（含生成 + LLM-as-judge，每条约 5~15 秒）
  python src/11_rag_eval.py --full --config hybrid

  # 3) 把 rerank / 查询改写也纳入对比（很慢，见下方说明）
  python src/11_rag_eval.py --retrieval-only --all --include-llm

  # 附加参数
  --dataset eval/qa_dataset.json   评估集路径
  --kb default                     知识库 id
  --limit 5                        只跑前 5 条，快速试跑
  --no-judge                       关闭 LLM 评判，只算确定性指标
  --report out.json                保存逐条明细

关于速度的重要提醒：
--retrieval-only 只是跳过「生成」，并不代表不调用大模型。
rerank 和查询改写发生在检索阶段，其中 rerank 的调用次数是
「样本数 × fetch_k」——16 条样本 × 20 候选 = 320 次推理，本地模型下要几十分钟。
所以这类预设默认不进 --all，需要显式加 --include-llm，建议配合 --limit 使用。

为什么必须有这一步：
RAG 的可调项（chunk_size / 召回通道 / 融合方式 / top_k / 是否重排 / 是否改写）
组合数量巨大，靠人肉试几个问题根本分不出好坏，还很容易被个别样本误导。
先用检索层指标定位瓶颈，再用端到端指标确认收益，这是唯一可靠的路径。

指标怎么读：
  hit@k 低       → 召回没捞到，改索引/切分/加 BM25 通道，别动提示词
  hit@k 高 mrr 低 → 捞到了但排得靠后，加重排
  检索层都好但 faithfulness 低 → 才是提示词或模型的问题
  abstain_rate 过高 → 门槛太严；过低（尤其库外问题不拒答）→ 幻觉风险
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 评估动辄跑几分钟。stdout 被重定向或接管道时默认是块缓冲，
# 进度会一直攒着不输出，看起来像卡死。这里强制行缓冲。
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).resolve().parent.parent

from config import get_scoring_llm  # noqa: E402
from rag import RagConfig, RagPipeline, ensure_kb  # noqa: E402
from rag.evaluate import (  # noqa: E402
    compare_configs,
    evaluate,
    load_dataset,
    render_table,
    save_report,
)

# 纯检索预设：只调 embedding，不调 LLM。16 条评估集通常几秒内跑完
CHEAP_PRESETS = {
    "vector": RagConfig(mode="vector", top_k=4),
    "bm25": RagConfig(mode="bm25", top_k=4),
    "mmr": RagConfig(mode="mmr", top_k=4),
    "hybrid": RagConfig(mode="hybrid", top_k=4, fetch_k=20),
    "hybrid_weighted": RagConfig(mode="hybrid", fusion="weighted", top_k=4, fetch_k=20),
}

# 依赖 LLM 的预设。注意它们在检索阶段就要调用大模型，
# 因此即便加了 --retrieval-only 也不会变快：
#   rerank 的调用次数 = 样本数 × fetch_k（16 × 20 = 320 次！）
#   multi_query      = 样本数 × 1
# 所以默认不进 --all，必须显式加 --include-llm。
LLM_PRESETS = {
    "hybrid_rerank": RagConfig(mode="hybrid", top_k=4, fetch_k=20, use_rerank=True),
    "hybrid_mq": RagConfig(mode="hybrid", top_k=4, fetch_k=20, use_multi_query=True),
    "hybrid_hyde": RagConfig(mode="hybrid", top_k=4, fetch_k=20, use_hyde=True),
}

PRESETS = {**CHEAP_PRESETS, **LLM_PRESETS}


def parse_args():
    p = argparse.ArgumentParser(description="RAG 离线评估")
    p.add_argument("--dataset", default=str(BASE_DIR / "eval" / "qa_dataset.json"))
    p.add_argument("--kb", default="default")
    p.add_argument("--config", default="hybrid", choices=list(PRESETS))
    p.add_argument("--all", action="store_true", help="对比全部纯检索预设（不含 rerank/改写）")
    p.add_argument("--include-llm", action="store_true",
                   help="把依赖 LLM 的预设(rerank/multi-query/hyde)一起纳入对比，会慢很多")
    p.add_argument("--retrieval-only", action="store_true", help="只评检索层，不调用生成")
    p.add_argument("--full", action="store_true", help="端到端评估（含生成）")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条样本（快速试跑用）")
    p.add_argument("--top-k", type=int, default=0,
                   help="覆盖预设的 top_k。库很小时 k 偏大会让所有策略都满分，"
                        "调到 1~2 才能看出区分度")
    p.add_argument("--no-judge", action="store_true", help="关闭 LLM-as-judge")
    p.add_argument("--report", default="", help="逐条明细输出路径（JSON）")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.retrieval_only and not args.full:
        args.retrieval_only = True  # 默认走快模式

    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    print(f"评估集: {args.dataset}（{len(cases)} 条）")
    ensure_kb(args.kb)

    # judge 也只需输出一个 1-5 的整数，用关闭思维链的实例，否则每条样本要多等几十秒
    judge_llm = None if (args.no_judge or args.retrieval_only) else get_scoring_llm()
    if judge_llm:
        print("LLM-as-judge: 开启（注意本地单模型自评会偏高）\n")

    if args.all:
        selected = dict(CHEAP_PRESETS)
        if args.include_llm:
            selected.update(LLM_PRESETS)
            print(f"已纳入 LLM 预设，预计需要约 {len(cases) * 20} 次额外模型调用，请耐心等待\n")
        configs = list(selected.values())
    else:
        configs = [PRESETS[args.config]]

    for c in configs:
        c.kb_id = args.kb
        if args.top_k:
            c.top_k = args.top_k

    if args.all or len(configs) > 1:
        rows = compare_configs(configs, cases, judge_llm, retrieval_only=args.retrieval_only)
        print("\n" + "=" * 78)
        print("A/B 对比结果")
        print("=" * 78)
        print(render_table(rows))
        _print_reading_guide(rows)
        return

    cfg = configs[0]
    pipeline = RagPipeline(cfg)

    if args.retrieval_only:
        rows = compare_configs(configs, cases, None, retrieval_only=True)
        print("\n" + render_table(rows))
        _print_reading_guide(rows)
        return

    print(f">>> 端到端评估: {cfg.label()}")
    report = evaluate(pipeline, cases, judge_llm)
    print("\n" + "=" * 78)
    print(render_table([report["summary"]]))

    print("\n逐条明细（拒答与低覆盖的样本最值得看）:")
    for r in report["cases"]:
        flag = "拒答" if r.abstained else f"cov={r.keyword_coverage}"
        print(f"  hit={r.hit:.0f} mrr={r.mrr:.2f} {flag:>10}  {r.question[:44]}")

    if args.report:
        save_report(report, args.report)
        print(f"\n明细已写入 {args.report}")
    _print_reading_guide([report["summary"]])


def _print_reading_guide(rows) -> None:
    print("\n" + "-" * 78)
    print("怎么读这张表：")
    print("  hit@k 低          → 召回阶段就没捞到，先改切分/加 BM25 通道，别调提示词")
    print("  hit@k 高但 mrr 低 → 捞到了排得靠后，加重排最有效")
    print("  ndcg 提升但延迟翻倍 → 权衡问题，看业务能接受多少延迟")
    print("  库外问题未拒答     → abstain_rate 偏低，说明幻觉风险高")
    print("-" * 78)


if __name__ == "__main__":
    main()
