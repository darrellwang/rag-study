"""
混合检索器：把召回、融合、重排、查询改写组装成一个可配置的整体。

完整流程（每一步都可关）：

    question
       │
       ├─(可选) multi_query / HyDE 改写 → 得到多条检索 query
       │
       ├─ 向量通道：query 向量化，取 fetch_k 条
       ├─ BM25 通道：query 分词，取 fetch_k 条
       │
       ├─ RRF 融合（只看名次，跨通道量纲无关）
       │
       ├─(可选) LLM 重排 + 相关性门槛
       │
       └─ top_k 条 → 交给生成

BM25 索引是懒加载的：第一次用到时把全库 chunk 拉进内存建索引。
知识库变更后要调 invalidate()，否则新文档搜不到。api.py 里入库后会调用。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.documents import Document

from config import FETCH_K, RRF_K, TOP_K, VECTOR_WEIGHT
from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion, weighted_score_fusion
from .query_transform import hyde_document, multi_query
from .rerank import llm_rerank
from .store import KnowledgeBase


@dataclass
class RetrievalResult:
    """检索结果 + 过程信息。过程信息是调优时唯一可靠的抓手。"""

    documents: List[Document] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    mode: str = "hybrid"
    queries: List[str] = field(default_factory=list)
    candidates: int = 0
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def table(self, width: int = 70) -> str:
        """把结果排成一张便于肉眼比对的表。"""
        if not self.documents:
            return "  (无命中)"
        lines = []
        for i, (d, s) in enumerate(zip(self.documents, self.scores), 1):
            src = d.metadata.get("file_name") or d.metadata.get("source", "?")
            body = d.page_content.replace("\n", " ")[:width]
            lines.append(f"  [{i}] score={s:.4f} src={src}\n      {body}...")
        return "\n".join(lines)


def _doc_key(doc: Document) -> str:
    """跨通道去重用的稳定标识。

    必须只用「来源 + 正文」计算，不能用库内 _id。因为两个通道拿到的 Document
    来路不同：向量通道来自 similarity_search（metadata 里没有 _id），
    BM25 通道来自 all_documents（metadata 里有 _id）。
    一旦用 _id，同一个 chunk 在两个通道会算成两个不同的 key，
    融合后 top-k 里就会出现重复片段 —— 这恰好摧毁了混合检索的意义。
    """
    meta = doc.metadata or {}
    raw = f"{meta.get('source', '')}::{doc.page_content}".encode()
    return hashlib.sha1(raw).hexdigest()


class HybridRetriever:
    def __init__(
        self,
        kb: KnowledgeBase,
        mode: str = "hybrid",
        top_k: int = TOP_K,
        fetch_k: int = FETCH_K,
        vector_weight: float = VECTOR_WEIGHT,
        rrf_k: int = RRF_K,
        fusion: str = "rrf",
        use_rerank: bool = False,
        use_multi_query: bool = False,
        use_hyde: bool = False,
        llm=None,
        scoring_llm=None,
        metadata_filter: Optional[dict] = None,
    ):
        self.kb = kb
        self.mode = mode
        self.top_k = top_k
        self.fetch_k = max(fetch_k, top_k)
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.fusion = fusion
        self.use_rerank = use_rerank
        self.use_multi_query = use_multi_query
        self.use_hyde = use_hyde
        self.llm = llm
        # 重排/打分走独立实例：需要关掉思维链，否则慢一个数量级
        self._scoring_llm = scoring_llm
        self.metadata_filter = metadata_filter

        self._bm25: Optional[BM25Index] = None
        self._corpus: List[Document] = []

    # ---------- BM25 懒加载 ----------
    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        self._corpus = self.kb.all_documents()
        if self.metadata_filter:
            self._corpus = [
                d for d in self._corpus
                if all(d.metadata.get(k) == v for k, v in self.metadata_filter.items())
            ]
        self._bm25 = BM25Index([d.page_content for d in self._corpus])

    def invalidate(self) -> None:
        """知识库写入/删除后必须调用，否则 BM25 索引是旧的。"""
        self._bm25 = None
        self._corpus = []

    @property
    def scoring_llm(self):
        """惰性创建打分模型：不开重排的场景不该为它付探测开销。"""
        if self._scoring_llm is None:
            from config import get_scoring_llm

            self._scoring_llm = get_scoring_llm()
        return self._scoring_llm

    # ---------- 单通道 ----------
    def _vector_channel(self, query: str, k: int) -> List[tuple]:
        """返回 [(Document, 相似度), ...]，相似度越大越相关。

        Chroma 返回的是距离，这里转成相似度，让所有通道的"越大越好"语义一致。
        """
        pairs = self.kb.similarity_search_with_score(query, k=k, metadata_filter=self.metadata_filter)
        return [(d, 1.0 / (1.0 + float(dist))) for d, dist in pairs]

    def _bm25_channel(self, query: str, k: int) -> List[tuple]:
        self._ensure_bm25()
        assert self._bm25 is not None
        return [(self._corpus[i], s) for i, s in self._bm25.search(query, k=k)]

    def _mmr_channel(self, query: str, k: int) -> List[tuple]:
        docs = self.kb.max_marginal_relevance_search(
            query, k=k, fetch_k=self.fetch_k, metadata_filter=self.metadata_filter
        )
        # MMR 不返回分值，用名次倒数作为占位分，保持接口一致
        return [(d, 1.0 / (i + 1)) for i, d in enumerate(docs)]

    # ---------- 主入口 ----------
    def search(self, question: str) -> RetrievalResult:
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}
        queries = [question]

        # 1. 查询改写
        if self.use_multi_query and self.llm is not None:
            t = time.perf_counter()
            queries = multi_query(question, self.llm, n=3)
            timings["multi_query"] = (time.perf_counter() - t) * 1000
        if self.use_hyde and self.llm is not None:
            t = time.perf_counter()
            queries.append(hyde_document(question, self.llm))
            timings["hyde"] = (time.perf_counter() - t) * 1000

        # 2. 多通道召回。每条 query 的每个通道各产出一个有序列表
        t = time.perf_counter()
        pool: Dict[str, Document] = {}
        ranked_lists: List[List[str]] = []
        scored_by_channel: Dict[str, Dict[str, float]] = {"vector": {}, "bm25": {}}

        def collect(pairs: List[tuple], channel: str) -> None:
            keys = []
            for d, s in pairs:
                key = _doc_key(d)
                pool[key] = d
                keys.append(key)
                prev = scored_by_channel[channel].get(key, 0.0)
                scored_by_channel[channel][key] = max(prev, s)
            ranked_lists.append(keys)

        for q in queries:
            if self.mode in ("vector", "hybrid"):
                collect(self._vector_channel(q, self.fetch_k), "vector")
            if self.mode in ("bm25", "hybrid"):
                collect(self._bm25_channel(q, self.fetch_k), "bm25")
            if self.mode == "mmr":
                collect(self._mmr_channel(q, self.fetch_k), "vector")
        timings["retrieve"] = (time.perf_counter() - t) * 1000

        candidates = len(pool)

        # 3. 融合
        if self.mode == "hybrid" and self.fusion == "weighted":
            fused = weighted_score_fusion(
                [
                    sorted(scored_by_channel["vector"].items(), key=lambda x: x[1], reverse=True),
                    sorted(scored_by_channel["bm25"].items(), key=lambda x: x[1], reverse=True),
                ],
                [self.vector_weight, 1.0 - self.vector_weight],
            )
        elif len(ranked_lists) > 1:
            fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
        else:
            single = ranked_lists[0] if ranked_lists else []
            channel = "bm25" if self.mode == "bm25" else "vector"
            fused = [(key, scored_by_channel[channel].get(key, 0.0)) for key in single]

        ordered = [(pool[k], s) for k, s in fused if k in pool]

        # 4. 重排：只对融合后的前 fetch_k 条做，控制 LLM 调用量
        if self.use_rerank and ordered:
            t = time.perf_counter()
            reranked = llm_rerank(
                question,
                [d for d, _ in ordered[: self.fetch_k]],
                self.scoring_llm,
                top_k=self.top_k,
            )
            timings["rerank"] = (time.perf_counter() - t) * 1000
            ordered = reranked

        final = ordered[: self.top_k]
        timings["total"] = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            documents=[d for d, _ in final],
            scores=[float(s) for _, s in final],
            mode=self.mode,
            queries=queries,
            candidates=candidates,
            timings_ms={k: round(v, 1) for k, v in timings.items()},
        )

    def invoke(self, question: str) -> List[Document]:
        """与 LangChain retriever 保持一致的调用形式。"""
        return self.search(question).documents

    def as_runnable(self):
        """包成 Runnable，可直接接进 `retriever | format_docs` 这类链。"""
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(self.invoke)
