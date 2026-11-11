"""
RAG 公共工具（兼容层）。

真正的实现已经拆进 `rag/` 包（切分 / 召回 / 融合 / 重排 / 评估各成一层）。
本模块只保留 03、04 等早期脚本依赖的函数签名，内部转调新实现，
这样老脚本不用改也能吃到新的切分策略、幂等入库和知识库枚举修复。

新代码请直接用：
    from rag import get_kb, HybridRetriever, RagPipeline, RagConfig

一处重要行为变更：默认知识库现在存放在 <CHROMA_PERSIST_DIR>/default/，
而不是直接存在 CHROMA_PERSIST_DIR 根目录。原来的做法让「根目录既是一个
Chroma 库、又是各知识库的父目录」，导致 Chroma 的内部段目录被误当成知识库。
"""
import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import DOCS_DIR
from rag import ChunkStrategy
from rag.store import (
    KnowledgeBase,
    delete_knowledge_base,
    get_kb,
    list_knowledge_bases,
)

# 内置示例知识库内容（data/ 为空时自动写入 data/example.txt）
SAMPLE_TEXT = (
    "LangChain 是一个用于开发由语言模型驱动的应用程序的框架，"
    "提供了链式调用、记忆、检索、代理等核心组件。\n"
    "LangGraph 是 LangChain 团队推出的库，用于构建有状态、多步骤、可循环的智能体工作流，"
    "通过图结构把多个节点和边组织起来。\n"
    "RAG（检索增强生成）通过先从知识库检索相关文档，再让大模型基于检索内容生成回答，从而减少幻觉。\n"
    "向量数据库（如 Chroma）将文本转化为嵌入向量并存储，支持基于语义相似度的检索。\n"
    "embedding 模型负责把文本转成高维向量，使语义相近的文本在向量空间中距离更近。\n"
    "temperature 参数控制生成随机性，值越低输出越确定；RAG 场景通常设为 0。\n"
    "文本切分（chunking）是 RAG 的关键预处理步骤，常用滑动窗口配合重叠来保留上下文。\n"
)


def _ensure_sample_docs() -> None:
    """data/ 目录为空时写入示例文档，保证首次运行有内容可检索。"""
    os.makedirs(DOCS_DIR, exist_ok=True)
    has_doc = any(
        name.lower().endswith((".txt", ".md", ".markdown", ".pdf"))
        for name in os.listdir(DOCS_DIR)
    )
    if not has_doc:
        path = os.path.join(DOCS_DIR, "example.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_TEXT)
        print(f"未发现文档，已生成示例文档: {path}")


def build_or_load_kb(kb_id: str = "default") -> Chroma:
    """构建或加载指定知识库，返回底层 Chroma（保持旧签名）。"""
    kb = get_kb(kb_id)
    if kb_id == "default" and kb.count() == 0:
        _ensure_sample_docs()
        print(f"构建默认知识库（文档目录: {DOCS_DIR}）...")
        for stats in kb.add_directory(DOCS_DIR, strategy=ChunkStrategy.MARKDOWN):
            print("  " + stats.summary())
    return kb.vs


def build_or_load_vectorstore() -> Chroma:
    """默认知识库的向量库。03/04 脚本使用。"""
    return build_or_load_kb("default")


def format_docs(docs: List[Document]) -> str:
    """把多个文档拼成一个字符串作为上下文。

    新代码建议改用 rag.pipeline.build_context()：它会给片段编号并生成引用清单，
    让模型可以在答案里标注 [1][2]，可溯源性强得多。
    """
    return "\n\n".join(d.page_content for d in docs)


def kb_dir(kb_id: str) -> str:
    from rag.store import kb_dir as _kb_dir

    return _kb_dir(kb_id)


def list_kbs() -> List[str]:
    """列出知识库。已修复：不再把 Chroma 内部段目录误报为知识库。"""
    return list_knowledge_bases()


def delete_kb(kb_id: str) -> bool:
    return delete_knowledge_base(kb_id)


def tag_documents(docs: List[Document], kb_id: str, filename: str) -> List[Document]:
    """为 chunk 注入来源 metadata。"""
    for d in docs:
        meta = dict(d.metadata or {})
        meta["kb_id"] = kb_id
        meta["source"] = meta.get("source") or filename
        meta.setdefault("file_name", os.path.basename(str(meta["source"])))
        d.metadata = meta
    return docs


def sources_from_docs(docs: List[Document]) -> List[dict]:
    """从检索结果抽取结构化来源（去重，保留文件名与片段）。"""
    out, seen = [], set()
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("file_name") or meta.get("source", "未知")
        key = (src, d.page_content[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "source": str(src),
                "kb_id": meta.get("kb_id", ""),
                "snippet": d.page_content,
            }
        )
    return out


def get_knowledge_base(kb_id: str = "default") -> KnowledgeBase:
    """拿到功能完整的 KnowledgeBase 对象（入库去重、按来源删除、统计）。"""
    return get_kb(kb_id)
