"""
多格式文档加载。

原项目只支持 .txt / .pdf，实际知识库里 Markdown、CSV、JSON 同样常见。
这里统一入口，并保证每个 Document 的 metadata 至少带上：
  source     原始文件路径（检索结果做引用时要用）
  file_name  文件名
  file_type  扩展名

注意 PDF：PyPDFLoader 按页返回 Document，页码在 metadata["page"]。
这个页码要一直传到最终引用里，否则用户拿到答案没法回原文核对。
"""
from __future__ import annotations

import csv
import json
import os
from typing import List

from langchain_core.documents import Document

SUPPORTED_EXTS = (".txt", ".md", ".markdown", ".pdf", ".csv", ".json", ".jsonl")


def _read_text(path: str) -> str:
    # 知识库文件编码常不统一，utf-8 失败时退回 gbk，再不行就忽略坏字节
    for enc in ("utf-8", "gbk"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_csv(path: str) -> List[Document]:
    """每行一个 Document，用 "列名: 值" 的形式序列化，便于语义检索。"""
    docs: List[Document] = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for i, row in enumerate(csv.DictReader(f)):
            body = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
            if body.strip():
                docs.append(Document(page_content=body, metadata={"row": i}))
    return docs


def _load_json(path: str) -> List[Document]:
    """支持 JSON 数组 / 单对象 / JSONL。数组元素逐个成为 Document。"""
    text = _read_text(path)
    records: list = []
    if path.lower().endswith(".jsonl"):
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        records = parsed if isinstance(parsed, list) else [parsed]

    docs = []
    for i, rec in enumerate(records):
        body = json.dumps(rec, ensure_ascii=False, indent=2) if not isinstance(rec, str) else rec
        docs.append(Document(page_content=body, metadata={"record": i}))
    return docs


def load_file(path: str) -> List[Document]:
    """加载单个文件，返回带来源 metadata 的 Document 列表。"""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"不支持的文件类型: {ext}（支持 {', '.join(SUPPORTED_EXTS)}）")

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        docs = PyPDFLoader(path).load()
    elif ext == ".csv":
        docs = _load_csv(path)
    elif ext in (".json", ".jsonl"):
        docs = _load_json(path)
    else:
        docs = [Document(page_content=_read_text(path), metadata={})]

    for d in docs:
        meta = dict(d.metadata or {})
        meta["source"] = path
        meta["file_name"] = os.path.basename(path)
        meta["file_type"] = ext.lstrip(".")
        d.metadata = meta
    return [d for d in docs if d.page_content.strip()]


def load_directory(directory: str, recursive: bool = True) -> List[Document]:
    """加载目录下所有支持格式的文件。单个文件失败不中断整体加载。"""
    docs: List[Document] = []
    if not os.path.isdir(directory):
        return docs

    walker = os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]
    for root, _, files in walker:
        for name in sorted(files):
            if name.startswith("."):
                continue
            if not name.lower().endswith(SUPPORTED_EXTS):
                continue
            path = os.path.join(root, name)
            try:
                docs.extend(load_file(path))
            except Exception as e:  # 坏文件不该让整库入库失败
                print(f"  [warn] 跳过 {path}: {e}")
    return docs
