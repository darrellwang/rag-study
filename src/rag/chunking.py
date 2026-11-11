"""
文本切分策略。

切分是 RAG 里性价比最高的调优点：检索的最小单位就是 chunk，
chunk 划得不合理，后面无论换多强的模型都救不回来。

本模块提供三种策略，覆盖大部分实际场景：

  recursive  通用递归字符切分。按 段落→句子→字 逐级降级，
             适合无结构的纯文本。
  token      按 tokenizer 的 token 数切分。当下游模型上下文预算很紧、
             需要精确控制时使用（字符数与 token 数在中英混排时差异很大）。
  markdown   先按标题层级拆成语义块，再对超长块做二次递归切分，
             并把标题路径写进 metadata。技术文档/Wiki 首选。

关键细节：markdown 策略会把「H1 > H2 > H3」的标题路径拼进 chunk 正文开头。
这样即使检索命中的是一个孤立段落，模型也知道它属于哪一节——
这个小技巧对长文档的问答准确率提升非常明显。
"""
from __future__ import annotations

from enum import Enum
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import CHUNK_SIZE, CHUNK_OVERLAP

# 中文文档的分隔符优先级：段落 > 换行 > 中文句末 > 中文分句 > 英文句末 > 空格 > 字
_ZH_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "，", " ", ""]

_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    TOKEN = "token"
    MARKDOWN = "markdown"


def _recursive_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_ZH_SEPARATORS,
        keep_separator=True,
    )


def _token_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """按 token 计数切分。chunk_size 此时的单位是 token 而非字符。

    tiktoken 首次使用需联网下载编码表。本项目定位是全离线本地部署，
    因此下载失败时降级为字符切分，而不是让整个入库流程失败。
    """
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_ZH_SEPARATORS,
        )
    except Exception as e:
        print(f"  [warn] tiktoken 编码表不可用（{type(e).__name__}），token 策略降级为字符切分")
        return _recursive_splitter(chunk_size, chunk_overlap)


def _heading_path(meta: dict) -> str:
    """从 MarkdownHeaderTextSplitter 写入的 metadata 还原标题路径。"""
    parts = [meta[key] for _, key in _MD_HEADERS if meta.get(key)]
    return " > ".join(parts)


def _split_markdown(
    docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS, strip_headers=False
    )
    tail_splitter = _recursive_splitter(chunk_size, chunk_overlap)

    out: List[Document] = []
    for doc in docs:
        sections = header_splitter.split_text(doc.page_content)
        for section in sections:
            merged_meta = {**(doc.metadata or {}), **(section.metadata or {})}
            path = _heading_path(merged_meta)
            if path:
                merged_meta["heading_path"] = path

            for piece in tail_splitter.split_text(section.page_content):
                # 标题路径前置，让孤立片段自带定位信息
                content = f"[{path}]\n{piece}" if path else piece
                out.append(Document(page_content=content, metadata=dict(merged_meta)))
    return out


def split_documents(
    docs: List[Document],
    strategy: ChunkStrategy | str = ChunkStrategy.RECURSIVE,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """按指定策略切分文档，并给每个 chunk 编号写入 metadata。

    markdown 策略只对 .md/.markdown 文件生效，其他文件自动退回 recursive，
    避免把普通文本里的 "#" 误当成标题。
    """
    if not docs:
        return []

    strategy = ChunkStrategy(strategy)

    if strategy is ChunkStrategy.MARKDOWN:
        md_docs = [d for d in docs if str(d.metadata.get("source", "")).lower().endswith((".md", ".markdown"))]
        other_docs = [d for d in docs if d not in md_docs]
        chunks = _split_markdown(md_docs, chunk_size, chunk_overlap)
        if other_docs:
            chunks += _recursive_splitter(chunk_size, chunk_overlap).split_documents(other_docs)
    elif strategy is ChunkStrategy.TOKEN:
        chunks = _token_splitter(chunk_size, chunk_overlap).split_documents(docs)
    else:
        chunks = _recursive_splitter(chunk_size, chunk_overlap).split_documents(docs)

    # 过滤空白 chunk（切分边界上偶尔会产出只有分隔符的碎片）
    chunks = [c for c in chunks if c.page_content.strip()]

    for i, c in enumerate(chunks):
        meta = dict(c.metadata or {})
        meta["chunk_index"] = i
        meta["chunk_strategy"] = strategy.value
        c.metadata = meta
    return chunks


def describe_chunks(chunks: List[Document]) -> dict:
    """统计切分结果，用于对比不同策略的粒度分布。"""
    if not chunks:
        return {"count": 0}
    lengths = sorted(len(c.page_content) for c in chunks)
    n = len(lengths)
    return {
        "count": n,
        "total_chars": sum(lengths),
        "min": lengths[0],
        "p50": lengths[n // 2],
        "p95": lengths[min(n - 1, int(n * 0.95))],
        "max": lengths[-1],
        "avg": round(sum(lengths) / n, 1),
    }
