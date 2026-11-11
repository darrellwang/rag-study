"""
GitHub 连接器。

真实模式：GitHub REST API（推荐用 PAT 鉴权）
  GET /repos/{owner}/{repo}/contents/{path}
  GET /repos/{owner}/{repo}/pulls
  GET /repos/{owner}/{repo}/issues

环境变量（见 .env.example）：
  GITHUB_TOKEN        PAT（repo 范围读权限）
  GITHUB_REPOS        逗号分隔 "owner/repo" 列表
  GITHUB_BRANCH       默认 main
"""
import os
from typing import List, Optional

from langchain_core.documents import Document

from connectors import BaseConnector


class GitHubConnector(BaseConnector):
    source_type = "github"

    def __init__(
        self,
        token: str = "",
        repos: Optional[list] = None,
        branch: str = "",
        mock: bool = None,
    ):
        self.token = token or os.getenv("GITHUB_TOKEN", "")
        self.repos = repos or [
            r.strip() for r in os.getenv("GITHUB_REPOS", "").split(",") if r.strip()
        ]
        self.branch = branch or os.getenv("GITHUB_BRANCH", "main")
        self.mock = mock if mock is not None else not (self.token and self.repos)

    def _fetch_raw(self) -> List[Document]:
        if self.mock:
            return self._mock_docs()
        return self._real_docs()

    def _real_docs(self) -> List[Document]:
        import base64
        import httpx

        client = httpx.Client(timeout=30.0)
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}

        docs = []
        for repo in self.repos:
            # 拉取 README 与最近的 PR/Issue 作为知识来源
            docs += self._fetch_readme(client, headers, repo)
            docs += self._fetch_pulls(client, headers, repo)
        return docs

    def _fetch_readme(self, client, headers, repo) -> List[Document]:
        try:
            resp = client.get(
                f"https://api.github.com/repos/{repo}/readme",
                headers=headers, params={"ref": self.branch},
            )
            resp.raise_for_status()
            content = base64.b64decode(resp.json().get("content", "")).decode("utf-8", "ignore")
            return [Document(
                page_content=content,
                metadata={
                    "source": f"github:{repo}/README.md",
                    "source_id": f"github:{repo}:readme",
                    "title": f"{repo} README",
                    "repo": repo,
                },
            )]
        except Exception as e:
            print(f"[GitHubConnector] 读取 {repo} README 失败: {e}")
            return []

    def _fetch_pulls(self, client, headers, repo) -> List[Document]:
        out = []
        try:
            resp = client.get(
                f"https://api.github.com/repos/{repo}/pulls",
                headers=headers, params={"state": "all", "per_page": 20, "sort": "updated"},
            )
            resp.raise_for_status()
            for pr in resp.json():
                title = pr.get("title", "")
                body = pr.get("body") or ""
                num = pr.get("number", "")
                if not (title or body):
                    continue
                out.append(Document(
                    page_content=f"PR #{num}: {title}\n\n{body}",
                    metadata={
                        "source": f"github:{repo}/pull/{num}",
                        "source_id": f"github:{repo}:pr:{num}",
                        "title": f"PR #{num} {title}",
                        "repo": repo,
                    },
                ))
        except Exception as e:
            print(f"[GitHubConnector] 读取 {repo} PR 失败: {e}")
        return out

    def _mock_docs(self) -> List[Document]:
        print("[GitHubConnector] 使用 MOCK 模式（未配置凭据）")
        readme = (
            "# rag-study\n本地知识库学习项目，基于 LangChain + LangGraph + Chroma。\n"
            "## 快速开始\n1. 安装依赖；2. 启动 LMStudio；3. 运行 API 与 UI。\n"
            "## 架构\nFastAPI 提供 RAG 接口，Streamlit 提供对话界面，"
            "向量库按知识库维度隔离。"
        )
        pr = (
            "PR #42: 增加连接器框架\n为支持 Confluence/Jira/GitHub 入库，"
            "抽象出 BaseConnector，统一 source metadata 与增量更新。"
        )
        return [
            Document(
                page_content=readme,
                metadata={
                    "source": "github:mock/rag-study/README.md",
                    "source_id": "github:mock:readme",
                    "title": "rag-study README",
                    "repo": "mock/rag-study",
                },
            ),
            Document(
                page_content=pr,
                metadata={
                    "source": "github:mock/rag-study/pull/42",
                    "source_id": "github:mock:pr:42",
                    "title": "PR #42 增加连接器框架",
                    "repo": "mock/rag-study",
                },
            ),
        ]
