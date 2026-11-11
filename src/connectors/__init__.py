"""
外部系统连接器（connector）统一框架。

各连接器负责把在线系统的内容拉取成 LangChain Document，
并统一注入来源 metadata（source / source_type / source_id 等），
以便溯源与后续更新。

设计原则：
  - 统一接口：BaseConnector.fetch() -> List[Document]
  - 真实/占位双模式：MOCK 模式返回样例数据，便于先把整链路跑通；
    真实凭据通过 .env 注入后即可切换到真实 API。
  - 网络代理：外部系统通常走公司内网/公网，与 LMStudio 的 localhost
    绕过代理相反，这里使用系统默认代理（信任 *_PROXY）。
  - 增量更新：通过 source_id 去重，重复 fetch 不会重复入库（由调用方控制）。
"""
from typing import List, Optional

from langchain_core.documents import Document

from rag_utils import tag_documents


class BaseConnector:
    """所有连接器的抽象基类。"""

    #: 来源类型标识，写入 metadata["source_type"]
    source_type: str = "unknown"

    def fetch(self, kb_id: str) -> List[Document]:
        """拉取内容并打标，返回 Document 列表。"""
        docs = self._fetch_raw()
        # 统一归类来源：source_type + kb_id，source 用原始 source（已含系统名）
        docs = tag_documents(docs, kb_id, filename="")
        for d in docs:
            meta = dict(d.metadata or {})
            meta["source_type"] = self.source_type
            d.metadata = meta
        return docs

    def _fetch_raw(self) -> List[Document]:
        raise NotImplementedError


# ---- 连接器工厂：按名称实例化 ----
_CONNECTORS = {}


def register(name: str, cls):
    _CONNECTORS[name] = cls


def get_connector(connector_type: str, config: dict) -> BaseConnector:
    if connector_type not in _CONNECTORS:
        raise ValueError(
            f"未知连接器类型: {connector_type}，可选: {', '.join(_CONNECTORS)}"
        )
    return _CONNECTORS[connector_type](**config)


def available_connectors() -> List[str]:
    return sorted(_CONNECTORS)


# 注册内置连接器（import 时触发）
from connectors.confluence import ConfluenceConnector  # noqa: E402
from connectors.jira import JiraConnector  # noqa: E402
from connectors.github import GitHubConnector  # noqa: E402

register("confluence", ConfluenceConnector)
register("jira", JiraConnector)
register("github", GitHubConnector)
