"""
本地知识库（Chroma 封装）。

这一层解决原实现里几个会在真实使用中暴露的问题：

1. 重复入库。原来每次上传都无条件 add_documents，同一份文件传两次，
   库里就有两份完全相同的 chunk。检索时 top-k 被同一段内容占满，
   等于白白浪费上下文预算。
   解法：chunk id = sha1(kb_id + source + 正文)，入库前先查 id 是否存在，
   已存在则跳过。天然幂等，重复上传是安全操作。

2. 无法更新/删除。文档改了只能重建整库。
   解法：metadata 里记 source，提供 delete_by_source()，
   「先删后插」即为增量更新。

3. 知识库列表误识别。原 list_kbs() 把 CHROMA_PERSIST_DIR 下所有子目录
   都当成知识库，但 Chroma 自己会在同级建 UUID 命名的段目录
   （如 94f920a5-77ea-.../），于是 UI 里会冒出一堆假知识库。
   解法：只认包含 chroma.sqlite3 的子目录。

4. 依赖私有 API。原来用 vs._collection.count() 拿数量。
   解法：统一收口到 count() / stats()，上层不再接触私有属性。
"""
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_PERSIST_DIR, CHUNK_OVERLAP, CHUNK_SIZE, get_embeddings
from .chunking import ChunkStrategy, split_documents
from .loaders import load_directory, load_file

_KB_CACHE: Dict[str, "KnowledgeBase"] = {}
_CHROMA_MARKER = "chroma.sqlite3"


@dataclass
class IngestStats:
    """入库结果。skipped 用于让使用者直观看到幂等生效。"""

    source: str
    raw_docs: int = 0
    chunks: int = 0
    added: int = 0
    skipped: int = 0
    total_in_kb: int = 0
    strategy: str = ChunkStrategy.RECURSIVE.value
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.source}: 原始 {self.raw_docs} 篇 → 切分 {self.chunks} 块 "
            f"→ 新增 {self.added}，跳过重复 {self.skipped}，库内共 {self.total_in_kb} 块"
        )


def _safe_kb_id(kb_id: str) -> str:
    """校验 kb_id，防止 ../ 造成的路径穿越。"""
    if not kb_id or not kb_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("kb_id 只能包含字母、数字、下划线和连字符")
    return kb_id


def kb_dir(kb_id: str) -> str:
    return os.path.join(CHROMA_PERSIST_DIR, _safe_kb_id(kb_id))


def chunk_id(kb_id: str, source: str, content: str) -> str:
    """内容寻址的 chunk id：同 kb + 同来源 + 同正文 ⇒ 同 id ⇒ 幂等。"""
    raw = f"{kb_id}::{source}::{content}".encode()
    return hashlib.sha1(raw).hexdigest()


# langchain-chroma 的默认 collection 名。必须沿用它，不能自己起名：
# 旧版本用的是默认值，一旦改名，已有知识库的数据会读不到——
# Chroma 会在同一个目录里静默新建一个空 collection，表现为「库还在但内容没了」。
DEFAULT_COLLECTION = "langchain"


class KnowledgeBase:
    """一个知识库 = 一个独立的 Chroma 持久化目录。"""

    def __init__(self, kb_id: str = "default", embeddings=None):
        self.kb_id = _safe_kb_id(kb_id)
        self.persist_dir = kb_dir(self.kb_id)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.vs = Chroma(
            collection_name=DEFAULT_COLLECTION,
            persist_directory=self.persist_dir,
            embedding_function=embeddings or get_embeddings(),
        )

    # ---------- 写入 ----------
    def add_documents(
        self,
        docs: List[Document],
        strategy: ChunkStrategy | str = ChunkStrategy.RECURSIVE,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        source_label: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> IngestStats:
        """切分 → 打 metadata → 去重 → 写入。"""
        stats = IngestStats(
            source=source_label or "documents",
            raw_docs=len(docs),
            strategy=ChunkStrategy(strategy).value,
        )
        if not docs:
            stats.total_in_kb = self.count()
            return stats

        chunks = split_documents(docs, strategy, chunk_size, chunk_overlap)
        stats.chunks = len(chunks)

        ids: List[str] = []
        for c in chunks:
            meta = dict(c.metadata or {})
            meta["kb_id"] = self.kb_id
            meta["source"] = meta.get("source") or source_label or "unknown"
            if extra_metadata:
                meta.update(extra_metadata)
            # Chroma 的 metadata 只接受标量，list/dict 会直接报错
            c.metadata = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
            ids.append(chunk_id(self.kb_id, str(c.metadata["source"]), c.page_content))

        # 库内已存在的 + 本批内重复的，都要排除
        seen = self._existing_ids(ids)
        deduped = []
        for i, c in zip(ids, chunks):
            if i in seen:
                continue
            seen.add(i)
            deduped.append((i, c))

        stats.skipped = len(chunks) - len(deduped)
        if deduped:
            self.vs.add_documents([c for _, c in deduped], ids=[i for i, _ in deduped])
            stats.added = len(deduped)

        stats.total_in_kb = self.count()
        return stats

    def add_file(self, path: str, strategy: ChunkStrategy | str = ChunkStrategy.RECURSIVE, **kw) -> IngestStats:
        docs = load_file(path)
        return self.add_documents(docs, strategy, source_label=os.path.basename(path), **kw)

    def add_directory(self, directory: str, strategy: ChunkStrategy | str = ChunkStrategy.MARKDOWN) -> List[IngestStats]:
        """按文件逐个入库，这样单个文件的失败/去重情况可以分别汇报。"""
        results: List[IngestStats] = []
        docs = load_directory(directory)
        by_source: Dict[str, List[Document]] = {}
        for d in docs:
            by_source.setdefault(str(d.metadata.get("source", "unknown")), []).append(d)

        for source, group in sorted(by_source.items()):
            results.append(
                self.add_documents(group, strategy, source_label=os.path.basename(source))
            )
        return results

    def _existing_ids(self, ids: List[str]) -> set:
        """查询哪些 id 已在库中。分批查避免超长 SQL 参数。"""
        found: set = set()
        for i in range(0, len(ids), 200):
            batch = ids[i : i + 200]
            try:
                got = self.vs.get(ids=batch, include=[])
                found.update(got.get("ids") or [])
            except Exception:
                # get 不可用时退化为不去重，宁可重复也不要阻塞入库
                return set()
        return found

    # ---------- 读取 ----------
    def count(self) -> int:
        try:
            return len(self.vs.get(include=[]).get("ids") or [])
        except Exception:
            return 0

    def all_documents(self) -> List[Document]:
        """取出全部 chunk。BM25 索引与统计需要遍历全库。"""
        try:
            got = self.vs.get(include=["documents", "metadatas"])
        except Exception:
            return []
        texts = got.get("documents") or []
        metas = got.get("metadatas") or []
        ids = got.get("ids") or []
        out = []
        for i, text in enumerate(texts):
            meta = dict(metas[i] or {}) if i < len(metas) else {}
            if i < len(ids):
                meta["_id"] = ids[i]
            out.append(Document(page_content=text, metadata=meta))
        return out

    def list_sources(self) -> Dict[str, int]:
        """来源文件 → chunk 数量。"""
        counts: Dict[str, int] = {}
        for d in self.all_documents():
            src = str(d.metadata.get("source", "unknown"))
            counts[src] = counts.get(src, 0) + 1
        return dict(sorted(counts.items()))

    def stats(self) -> dict:
        docs = self.all_documents()
        lengths = sorted(len(d.page_content) for d in docs)
        sources: Dict[str, int] = {}
        for d in docs:
            src = str(d.metadata.get("source", "unknown"))
            sources[src] = sources.get(src, 0) + 1
        n = len(lengths)
        return {
            "kb_id": self.kb_id,
            "chunks": n,
            "sources": sources,
            "total_chars": sum(lengths),
            "avg_chunk_chars": round(sum(lengths) / n, 1) if n else 0,
            "max_chunk_chars": lengths[-1] if n else 0,
            "persist_dir": self.persist_dir,
        }

    def similarity_search_with_score(
        self, query: str, k: int = 4, metadata_filter: Optional[dict] = None
    ) -> List[tuple]:
        """向量检索，返回 (Document, 距离)。距离越小越相似。"""
        return self.vs.similarity_search_with_score(query, k=k, filter=metadata_filter)

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5,
        metadata_filter: Optional[dict] = None,
    ) -> List[Document]:
        """MMR 检索：在相关性和多样性之间取平衡，抑制近重复片段挤占 top-k。"""
        return self.vs.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=metadata_filter
        )

    # ---------- 删除 ----------
    def delete_by_source(self, source: str) -> int:
        """删除某个来源的全部 chunk，返回删除数量。用于文档更新与下架。"""
        try:
            got = self.vs.get(where={"source": source}, include=[])
        except Exception:
            return 0
        ids = got.get("ids") or []
        if ids:
            self.vs.delete(ids=ids)
        return len(ids)

    def clear(self) -> int:
        """清空知识库内容但保留知识库本身。"""
        ids = self.vs.get(include=[]).get("ids") or []
        if ids:
            self.vs.delete(ids=ids)
        return len(ids)


def get_kb(kb_id: str = "default", embeddings=None) -> KnowledgeBase:
    """带进程内缓存的知识库获取（避免重复打开 Chroma 连接）。"""
    if kb_id not in _KB_CACHE:
        _KB_CACHE[kb_id] = KnowledgeBase(kb_id, embeddings)
    return _KB_CACHE[kb_id]


def ensure_kb(kb_id: str = "default", docs_dir: Optional[str] = None, verbose: bool = True) -> KnowledgeBase:
    """获取知识库；若为空则自动从 docs_dir 导入，保证示例脚本开箱可跑。"""
    from config import DOCS_DIR

    kb = get_kb(kb_id)
    if kb.count() == 0:
        directory = docs_dir or DOCS_DIR
        if verbose:
            print(f"知识库 `{kb_id}` 为空，正在导入 {directory} ...")
        for stats in kb.add_directory(directory):
            if verbose:
                print("  " + stats.summary())
    if verbose:
        print(f"知识库 `{kb_id}`: {kb.count()} 个 chunk\n")
    return kb


def list_knowledge_bases() -> List[str]:
    """列出真实知识库：只认含 chroma.sqlite3 的子目录，排除 Chroma 内部段目录。"""
    if not os.path.isdir(CHROMA_PERSIST_DIR):
        return []
    return sorted(
        name
        for name in os.listdir(CHROMA_PERSIST_DIR)
        if os.path.isfile(os.path.join(CHROMA_PERSIST_DIR, name, _CHROMA_MARKER))
    )


def delete_knowledge_base(kb_id: str) -> bool:
    """删除整个知识库目录，返回是否真实存在过。

    坑点：chromadb 在进程内按持久化路径全局缓存 client（SharedSystemClient）。
    若只 rmtree 目录而不清缓存，之后重建同名知识库会拿到指向已删除 sqlite 文件的
    旧句柄，写入时报 "attempt to write a readonly database"。
    所以删除前必须清掉 chroma 的 system 缓存。
    """
    _KB_CACHE.pop(kb_id, None)
    _reset_chroma_client_cache()
    d = kb_dir(kb_id)
    if os.path.isdir(d):
        shutil.rmtree(d)
        return True
    return False


def _reset_chroma_client_cache() -> None:
    """清空 chromadb 的进程内 client 缓存。版本差异下失败也不影响主流程。"""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
