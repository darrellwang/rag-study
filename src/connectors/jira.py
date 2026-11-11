"""
Jira 连接器。

真实模式：Jira Cloud REST API
  GET /rest/api/3/search?jql=...
使用 Basic Auth（email:API_TOKEN）或 PAT（Bearer）。

环境变量（见 .env.example）：
  JIRA_BASE_URL   e.g. https://your-org.atlassian.net
  JIRA_EMAIL      登录邮箱
  JIRA_API_TOKEN  API Token / PAT
  JIRA_JQL        自定义 JQL，默认 "ORDER BY updated DESC"（取前 50 条）
"""
import os
from typing import List, Optional

from langchain_core.documents import Document

from connectors import BaseConnector


class JiraConnector(BaseConnector):
    source_type = "jira"

    def __init__(
        self,
        base_url: str = "",
        email: str = "",
        api_token: str = "",
        jql: str = "",
        mock: bool = None,
    ):
        self.base_url = base_url or os.getenv("JIRA_BASE_URL", "")
        self.email = email or os.getenv("JIRA_EMAIL", "")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN", "")
        self.jql = jql or os.getenv("JIRA_JQL", "ORDER BY updated DESC")
        self.mock = mock if mock is not None else not (self.base_url and self.api_token)

    def _fetch_raw(self) -> List[Document]:
        if self.mock:
            return self._mock_docs()
        return self._real_docs()

    def _real_docs(self) -> List[Document]:
        import httpx

        client = httpx.Client(timeout=30.0)
        auth = (self.email, self.api_token) if self.email else None
        headers = {"Accept": "application/json"}
        if not auth:
            headers["Authorization"] = f"Bearer {self.api_token}"

        resp = client.get(
            f"{self.base_url.rstrip('/')}/rest/api/3/search",
            params={"jql": self.jql, "maxResults": 50, "fields": "summary,description,comment"},
            auth=auth, headers=headers,
        )
        resp.raise_for_status()

        docs = []
        for issue in resp.json().get("issues", []):
            key = issue.get("key", "")
            fields = issue.get("fields", {})
            summary = fields.get("summary", "")
            desc = fields.get("description", "") or ""
            desc = _adf_to_text(desc)
            comments = [
                _adf_to_text(c.get("body", ""))
                for c in fields.get("comment", {}).get("comments", [])
            ]
            text = f"标题: {summary}\n\n描述: {desc}"
            if comments:
                text += "\n\n评论:\n" + "\n".join(f"- {c}" for c in comments if c)
            if not text.strip():
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": f"jira:{self.base_url}/browse/{key}",
                    "source_id": f"jira:{key}",
                    "title": summary,
                    "issue_key": key,
                },
            ))
        return docs

    def _mock_docs(self) -> List[Document]:
        print("[JiraConnector] 使用 MOCK 模式（未配置凭据）")
        sample = [
            ("LOG-1024", "登录接口偶发 500",
             "用户在高并发下登录偶发 HTTP 500，怀疑连接池耗尽。"
             "已扩容数据库连接池至 200，并增加熔断；后续观察监控确认。"),
            ("PAY-331", "支付回调超时", "第三方支付回调平均耗时 3s 超过网关 2s 限制，"
             "已调整网关超时并增加异步重试队列，待灰度验证。"),
            ("OPS-77", "磁盘告警处理", "prod-03 磁盘使用率超 85%，清理过期日志后恢复。"
             "后续接入自动轮转策略避免复发。"),
        ]
        return [
            Document(
                page_content=f"标题: {title}\n\n描述: {body}",
                metadata={
                    "source": f"jira:mock/{key}",
                    "source_id": f"jira:{key}",
                    "title": title,
                    "issue_key": key,
                },
            )
            for key, title, body in sample
        ]


def _adf_to_text(adf) -> str:
    """把 Jira 的 Atlassian Document Format（ADF）粗略转成文本。"""
    if isinstance(adf, str):
        return adf
    if not isinstance(adf, dict):
        return ""
    parts = []

    def walk(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "text":
            parts.append(node.get("text", ""))
        for child in node.get("content", []):
            walk(child)

    walk(adf)
    return "\n".join(p for p in parts if p).strip()
