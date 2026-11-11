"""
Confluence 连接器。

真实模式：通过 Confluence Cloud REST API
  GET /rest/api/content?expand=body.view&limit=...
使用 Basic Auth（email:API_TOKEN）或 PAT（Bearer）。

由于当前阶段先 mock，默认从环境变量读凭据；若凭据为空则进入
MOCK 模式返回样例文档，保证整条入库链路可跑通。

环境变量（见 .env.example）：
  CONFLUENCE_BASE_URL  e.g. https://your-org.atlassian.net/wiki
  CONFLUENCE_EMAIL     登录邮箱（Basic Auth 用户名）
  CONFLUENCE_API_TOKEN API Token / PAT
  CONFLUENCE_SPACES    逗号分隔的 space key 列表，如 "DEV,OPS"
"""
import os
from typing import List, Optional

from langchain_core.documents import Document

from connectors import BaseConnector


class ConfluenceConnector(BaseConnector):
    source_type = "confluence"

    def __init__(
        self,
        base_url: str = "",
        email: str = "",
        api_token: str = "",
        spaces: Optional[list] = None,
        mock: bool = None,
    ):
        self.base_url = base_url or os.getenv("CONFLUENCE_BASE_URL", "")
        self.email = email or os.getenv("CONFLUENCE_EMAIL", "")
        self.api_token = api_token or os.getenv("CONFLUENCE_API_TOKEN", "")
        self.spaces = spaces or [
            s.strip() for s in os.getenv("CONFLUENCE_SPACES", "").split(",") if s.strip()
        ]
        # 未显式指定且缺少凭据时自动 mock
        self.mock = mock if mock is not None else not (self.base_url and self.api_token)

    def _fetch_raw(self) -> List[Document]:
        if self.mock:
            return self._mock_docs()
        return self._real_docs()

    def _real_docs(self) -> List[Document]:
        import httpx

        # 外部系统走公司内网/公网，使用系统默认代理（信任 *_PROXY）
        client = httpx.Client(timeout=30.0)
        auth = (self.email, self.api_token) if self.email else None
        headers = {"Accept": "application/json"}
        if not auth:
            # PAT 模式
            headers["Authorization"] = f"Bearer {self.api_token}"

        docs = []
        space_keys = self.spaces or [""]
        for space in space_keys:
            params = {"expand": "body.view", "limit": 50}
            if space:
                params["spaceKey"] = space
            resp = client.get(
                f"{self.base_url.rstrip('/')}/rest/api/content",
                params=params, auth=auth, headers=headers,
            )
            resp.raise_for_status()
            for item in resp.json().get("results", []):
                title = item.get("title", "untitled")
                page_id = item.get("id", "")
                body = (
                    item.get("body", {})
                    .get("view", {})
                    .get("value", "")
                )
                text = _strip_html(body)
                if not text.strip():
                    continue
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": f"confluence:{self.base_url}/{page_id}",
                        "source_id": f"confluence:{page_id}",
                        "title": title,
                        "space": space,
                    },
                ))
        return docs

    def _mock_docs(self) -> List[Document]:
        print("[ConfluenceConnector] 使用 MOCK 模式（未配置凭据）")
        sample = [
            ("上线发布流程",
             "生产环境发布需走灰度发布：先在预发验证，再按 10% -> 50% -> 100% "
             "比例放量，每阶段观察 30 分钟核心指标（错误率、P99 延迟）。"),
            ("事故响应手册",
             "P0 事故需在 15 分钟内拉起作战室，指定指挥（IC）与通信负责人，"
             "每 30 分钟同步一次进展，恢复后 3 个工作日内提交复盘报告。"),
            ("代码评审规范",
             "MR 至少需要 1 名 Reviewer 批准；涉及数据库变更必须包含回滚脚本；"
             "禁止在业务高峰期合并高风险变更。"),
        ]
        return [
            Document(
                page_content=body,
                metadata={
                    "source": f"confluence:mock/{i}",
                    "source_id": f"confluence:mock/{i}",
                    "title": title,
                    "space": "MOCK",
                },
            )
            for i, (title, body) in enumerate(sample)
        ]


def _strip_html(html: str) -> str:
    """极简 HTML -> 文本（避免引入额外依赖）。"""
    import re

    text = re.sub(r"<script.*?</script>", "", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
