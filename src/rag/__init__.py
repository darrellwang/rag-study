"""
RAG 核心能力包（教学向实现，尽量少依赖、多注释）。

模块划分对应 RAG 的四个可独立优化的环节：

    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ 索引     │→ │ 召回     │→ │ 精排     │→ │ 生成     │
    │ loaders  │   │ bm25     │   │ rerank   │   │ pipeline │
    │ chunking │   │ retriever│   │ fusion   │   │          │
    │ store    │   │ query_*  │   │          │   │          │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘
                          ↓ 全链路可量化
                       evaluate

工程经验：RAG 效果不好时，八成问题出在「索引 + 召回」而不是提示词。
所以本包把这两层的每个选择（切分粒度、召回通道、融合方式、重排）都做成
可开关的参数，配合 evaluate 模块做 A/B，而不是靠感觉调。
"""
import sys
from pathlib import Path

# 保证从任意 cwd（项目根 / src / 容器内）都能 import config
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from .chunking import ChunkStrategy, split_documents, describe_chunks  # noqa: E402
from .loaders import load_file, load_directory, SUPPORTED_EXTS  # noqa: E402
from .bm25 import BM25Index, tokenize  # noqa: E402
from .fusion import reciprocal_rank_fusion, weighted_score_fusion  # noqa: E402
from .rerank import llm_rerank  # noqa: E402
from .query_transform import multi_query, hyde_document, decompose_question  # noqa: E402
from .retriever import HybridRetriever, RetrievalResult  # noqa: E402
from .store import (  # noqa: E402
    KnowledgeBase,
    IngestStats,
    get_kb,
    ensure_kb,
    list_knowledge_bases,
    delete_knowledge_base,
)
from .pipeline import RagPipeline, RagConfig, RagAnswer  # noqa: E402

__all__ = [
    "ChunkStrategy",
    "split_documents",
    "describe_chunks",
    "load_file",
    "load_directory",
    "SUPPORTED_EXTS",
    "BM25Index",
    "tokenize",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
    "llm_rerank",
    "multi_query",
    "hyde_document",
    "decompose_question",
    "HybridRetriever",
    "RetrievalResult",
    "KnowledgeBase",
    "IngestStats",
    "get_kb",
    "ensure_kb",
    "list_knowledge_bases",
    "delete_knowledge_base",
    "RagPipeline",
    "RagConfig",
    "RagAnswer",
]
