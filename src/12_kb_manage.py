"""
12 - 本地知识库全生命周期管理 CLI

之前只能通过 HTTP 接口或 UI 操作知识库，调试很不方便。
这个命令行工具覆盖知识库的完整生命周期，也是理解「知识库不只是入库」的入口。

用法:
  python src/12_kb_manage.py ls                                  列出所有知识库
  python src/12_kb_manage.py stats  [--kb default]               统计信息
  python src/12_kb_manage.py sources [--kb default]              各来源的 chunk 数
  python src/12_kb_manage.py add data/rag_retrieval.md           入库单个文件
  python src/12_kb_manage.py add data/ --strategy markdown       入库整个目录
  python src/12_kb_manage.py search "BM25 参数" --mode hybrid     检索调试
  python src/12_kb_manage.py ask "chunk 怎么切"                   完整问答（带引用）
  python src/12_kb_manage.py rm-source rag_retrieval.md          按来源删除
  python src/12_kb_manage.py update data/rag_retrieval.md        增量更新（先删后插）
  python src/12_kb_manage.py clear --kb demo                     清空内容保留库
  python src/12_kb_manage.py drop  --kb demo                     删除整个库
  python src/12_kb_manage.py chunk-compare data/rag_retrieval.md 对比三种切分策略

重复执行 add 是安全的：chunk id 由内容哈希决定，第二次会全部命中「跳过重复」。
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CHUNK_OVERLAP, CHUNK_SIZE, get_llm  # noqa: E402
from rag import (  # noqa: E402
    ChunkStrategy,
    HybridRetriever,
    RagConfig,
    RagPipeline,
    delete_knowledge_base,
    describe_chunks,
    get_kb,
    list_knowledge_bases,
    load_file,
    split_documents,
)


def cmd_ls(args):
    kbs = list_knowledge_bases()
    if not kbs:
        print("(还没有任何知识库，先用 add 命令入库)")
        return
    for name in kbs:
        kb = get_kb(name)
        print(f"  {name:<20} {kb.count():>6} chunks")


def cmd_stats(args):
    st = get_kb(args.kb).stats()
    print(f"知识库   : {st['kb_id']}")
    print(f"存储目录 : {st['persist_dir']}")
    print(f"chunk 数 : {st['chunks']}")
    print(f"总字符数 : {st['total_chars']}")
    print(f"平均长度 : {st['avg_chunk_chars']}  最长: {st['max_chunk_chars']}")
    print(f"来源数   : {len(st['sources'])}")


def cmd_sources(args):
    sources = get_kb(args.kb).list_sources()
    if not sources:
        print("(空库)")
        return
    for src, n in sources.items():
        print(f"  {n:>5} chunks  {src}")


def cmd_add(args):
    kb = get_kb(args.kb)
    if os.path.isdir(args.path):
        for st in kb.add_directory(args.path, strategy=args.strategy):
            print("  " + st.summary())
    else:
        print("  " + kb.add_file(args.path, strategy=args.strategy).summary())


def cmd_update(args):
    """增量更新 = 先按来源删除，再重新入库。文档改动后用这个而不是 add。"""
    kb = get_kb(args.kb)
    name = os.path.basename(args.path)
    removed = kb.delete_by_source(name)
    print(f"  已删除来源 `{name}` 的 {removed} 个旧 chunk")
    print("  " + kb.add_file(args.path, strategy=args.strategy).summary())


def cmd_rm_source(args):
    removed = get_kb(args.kb).delete_by_source(args.source)
    print(f"  删除了 {removed} 个 chunk" if removed else f"  未找到来源 `{args.source}`")


def cmd_clear(args):
    print(f"  已清空 {get_kb(args.kb).clear()} 个 chunk（知识库保留）")


def cmd_drop(args):
    ok = delete_knowledge_base(args.kb)
    print(f"  知识库 `{args.kb}` 已删除" if ok else f"  知识库 `{args.kb}` 不存在")


def cmd_search(args):
    """只做检索不做生成。排查「答得不对」时第一步永远是先看检索到了什么。"""
    llm = get_llm(temperature=0) if args.rerank else None
    retriever = HybridRetriever(
        kb=get_kb(args.kb), mode=args.mode, top_k=args.k, use_rerank=args.rerank, llm=llm
    )
    result = retriever.search(args.query)
    print(f"模式 {result.mode} | 候选 {result.candidates} | 耗时 {result.timings_ms}")
    print(result.table(width=110))


def cmd_ask(args):
    cfg = RagConfig(kb_id=args.kb, mode=args.mode, top_k=args.k, use_rerank=args.rerank)
    result = RagPipeline(cfg).answer(args.query)
    print(result.pretty())


def cmd_chunk_compare(args):
    """同一份文件用三种策略切分，对比 chunk 数量与长度分布。"""
    docs = load_file(args.path)
    print(f"文件: {args.path}（{sum(len(d.page_content) for d in docs)} 字符）")
    print(f"参数: chunk_size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}\n")
    for strategy in (ChunkStrategy.RECURSIVE, ChunkStrategy.TOKEN, ChunkStrategy.MARKDOWN):
        chunks = split_documents(docs, strategy)
        print(f"  {strategy.value:<10} {describe_chunks(chunks)}")
    print("\n提示: token 策略的长度统计仍以字符计，所以数值看起来更大 —— "
          "它控制的是 token 数，中文一字约 1~2 token。")

    chunks = split_documents(docs, ChunkStrategy.MARKDOWN)
    if chunks:
        print(f"\nmarkdown 策略首个 chunk（注意开头被注入的标题路径）:\n{chunks[0].page_content[:260]}")


def build_parser():
    p = argparse.ArgumentParser(description="本地知识库管理")
    p.add_argument("--kb", default="default", help="知识库 id")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ls").set_defaults(func=cmd_ls)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    sub.add_parser("sources").set_defaults(func=cmd_sources)

    for name, func in (("add", cmd_add), ("update", cmd_update)):
        sp = sub.add_parser(name)
        sp.add_argument("path")
        sp.add_argument("--strategy", default="markdown", choices=[s.value for s in ChunkStrategy])
        sp.set_defaults(func=func)

    sp = sub.add_parser("rm-source")
    sp.add_argument("source", help="来源文件名（与 metadata.source 一致）")
    sp.set_defaults(func=cmd_rm_source)

    sub.add_parser("clear").set_defaults(func=cmd_clear)
    sub.add_parser("drop").set_defaults(func=cmd_drop)

    for name, func in (("search", cmd_search), ("ask", cmd_ask)):
        sp = sub.add_parser(name)
        sp.add_argument("query")
        sp.add_argument("--mode", default="hybrid", choices=["vector", "mmr", "bm25", "hybrid"])
        sp.add_argument("-k", type=int, default=4)
        sp.add_argument("--rerank", action="store_true")
        sp.set_defaults(func=func)

    sp = sub.add_parser("chunk-compare")
    sp.add_argument("path")
    sp.set_defaults(func=cmd_chunk_compare)
    return p


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
