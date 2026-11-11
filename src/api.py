"""
本地知识库服务 (FastAPI)

两套问答引擎，可按请求切换，便于对比：
  engine=pipeline （默认）混合检索 + 可选重排/查询改写 + 带引用生成 + 拒答
  engine=graph          LangGraph Corrective RAG（检索→评估→不相关则重写问题重试）

相比早期版本修掉的几个问题：
  · 每次 /query 都重建一次 LangGraph（因为要注入对话历史）→ 历史改为放进图的
    state，图本身可以按 kb 缓存复用
  · 文档相关性评估逐条串行调用大模型 → 改为批量并发
  · /query/stream 先算完整答案再切片，是假流式 → 改为真正的 token 级流式
  · 用 vs._collection.count() 这类私有 API → 收口到 KnowledgeBase
  · 知识库列表把 Chroma 内部段目录误报成知识库 → 只认含 chroma.sqlite3 的目录
  · 重复上传同一文件会产生重复 chunk → 内容哈希去重，入库幂等

接口:
  GET    /health                          健康检查 + 知识库列表
  POST   /kb                              创建知识库（登记 owner）
  GET    /kb                              列出知识库
  GET    /kb/{kb_id}/stats                统计信息（chunk 数、来源分布）
  GET    /kb/{kb_id}/sources              来源文件列表
  DELETE /kb/{kb_id}                      删除知识库
  DELETE /kb/{kb_id}/sources              按来源删除 chunk（文档下架/更新用）
  POST   /ingest                          上传文档入库（支持切分策略选择）
  POST   /ingest/connector                从 Confluence/Jira/GitHub 拉取入库
  GET    /connectors                      可用连接器
  POST   /search                          只检索不生成（排查「答得不对」的第一站）
  POST   /query                           问答（返回引用清单与各阶段耗时）
  POST   /query/stream                    流式问答（SSE，真 token 流）
  GET    /sessions/{sid}                  会话历史
  DELETE /sessions/{sid}                  清除会话

运行:
  source .venv/bin/activate
  uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
"""
# 保证从任意 cwd 启动都能以 src 为包根导入 config / rag
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import os
import tempfile
import time
import traceback
import uuid
from typing import Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from config import (
    FETCH_K,
    RETRIEVAL_MODE,
    SERVICE_API_KEY,
    TOP_K,
    USE_RERANK,
    get_llm,
    get_scoring_llm,
)
from connectors import available_connectors, get_connector
from kb_registry import delete_registry, is_owner, register_kb, user_id_from_key
from rag import (
    ChunkStrategy,
    HybridRetriever,
    RagConfig,
    RagPipeline,
    delete_knowledge_base,
    get_kb,
    list_knowledge_bases,
)
from rag.loaders import SUPPORTED_EXTS, load_file
from rag.pipeline import ANSWER_PROMPT, build_context
from session_store import append_message, delete_session, get_messages, history_text

app = FastAPI(title="本地知识库服务", version="0.6.0")

MAX_RETRIES = 2
HISTORY_TURNS = 6
GRADE_CONCURRENCY = 4

_llm = None
_scoring_llm = None
_pipeline_cache: Dict[Tuple, RagPipeline] = {}
_graph_cache: Dict[str, object] = {}


# ===== 鉴权 =====
def _auth(x_api_key: Optional[str] = Header(None)) -> str:
    """返回当前 user_id；若启用 Key 且不匹配则 401。"""
    if SERVICE_API_KEY and x_api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="缺少或错误的 API Key")
    return user_id_from_key(x_api_key or "")


def _require_write(kb_id: str, user: str) -> None:
    if not is_owner(kb_id, user):
        raise HTTPException(status_code=403, detail="无权写入该知识库")


# ===== 请求/响应模型 =====
class RetrievalOptions(BaseModel):
    """检索相关的可调项。全部有默认值，老客户端不传也能工作。"""

    mode: str = Field(default=RETRIEVAL_MODE, description="vector | mmr | bm25 | hybrid")
    k: int = Field(default=TOP_K, ge=1, le=20)
    fetch_k: int = Field(default=FETCH_K, ge=1, le=100)
    fusion: str = Field(default="rrf", description="rrf | weighted")
    use_rerank: bool = USE_RERANK
    use_multi_query: bool = False
    use_hyde: bool = False
    metadata_filter: Optional[dict] = None


class QueryRequest(RetrievalOptions):
    question: str
    kb_id: str = "default"
    session_id: Optional[str] = None
    engine: str = Field(default="pipeline", description="pipeline | graph")


class SearchRequest(RetrievalOptions):
    query: str
    kb_id: str = "default"


class CreateKbRequest(BaseModel):
    kb_id: Optional[str] = None


class IngestResponse(BaseModel):
    filename: str
    kb_id: str
    raw_docs: int
    chunks: int
    added: int
    skipped: int
    total_docs: int
    strategy: str


class SourceItem(BaseModel):
    index: Optional[int] = None
    source: str
    kb_id: str = ""
    heading_path: Optional[str] = None
    page: Optional[int] = None
    snippet: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    session_id: str
    abstained: bool = False
    timings_ms: dict = {}
    config: str = ""
    queries: List[str] = []


class HealthResponse(BaseModel):
    llm_ok: bool
    embedding_ok: bool
    kbs: List[str]


# ===== 内部工具 =====
def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(temperature=0)
    return _llm


def _get_scoring_llm():
    """打分/分类专用（关闭思维链）。相关性评估和重排都走这个，否则慢一个数量级。"""
    global _scoring_llm
    if _scoring_llm is None:
        _scoring_llm = get_scoring_llm()
    return _scoring_llm


def _config_from(req: RetrievalOptions, kb_id: str) -> RagConfig:
    return RagConfig(
        kb_id=kb_id,
        mode=req.mode,
        top_k=req.k,
        fetch_k=max(req.fetch_k, req.k),
        fusion=req.fusion,
        use_rerank=req.use_rerank,
        use_multi_query=req.use_multi_query,
        use_hyde=req.use_hyde,
        metadata_filter=req.metadata_filter,
    )


def _get_pipeline(cfg: RagConfig) -> RagPipeline:
    """按配置缓存 pipeline。BM25 索引建在 pipeline 里，缓存能省下重复建索引的开销。"""
    key = (
        cfg.kb_id, cfg.mode, cfg.top_k, cfg.fetch_k, cfg.fusion,
        cfg.use_rerank, cfg.use_multi_query, cfg.use_hyde,
        json.dumps(cfg.metadata_filter, sort_keys=True) if cfg.metadata_filter else None,
    )
    if key not in _pipeline_cache:
        _pipeline_cache[key] = RagPipeline(
            cfg, llm=_get_llm(), scoring_llm=_get_scoring_llm()
        )
    return _pipeline_cache[key]


def _invalidate(kb_id: str) -> None:
    """知识库内容变更后，让缓存的 BM25 索引和图失效。"""
    for key, pipeline in list(_pipeline_cache.items()):
        if key[0] == kb_id:
            pipeline.invalidate()
            _pipeline_cache.pop(key, None)
    _graph_cache.pop(kb_id, None)


def _to_source_items(citations: List[dict]) -> List[SourceItem]:
    return [SourceItem(**c) for c in citations]


# ===== LangGraph Corrective RAG =====
class GraphState(TypedDict):
    question: str
    original_question: str
    history: str          # 历史放进 state，图就不必为每个会话重建
    documents: list
    generation: str
    retry_count: int


_GRADE_PROMPT = ChatPromptTemplate.from_template(
    "你是一个评估文档相关性的助手。请判断下面的文档是否与问题相关。"
    "只回答 'yes' 或 'no'。\n\n问题: {question}\n文档: {document}\n判断:"
)

_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "请用更好的措辞重写下面的问题，使其更适合用于文档检索。"
    "只输出重写后的问题，不要解释。\n原问题: {question}\n重写后:"
)


def _graph_retrieve(state: GraphState, retriever: HybridRetriever) -> dict:
    return {"documents": retriever.invoke(state["question"])}


def _graph_grade(state: GraphState, llm) -> dict:
    """批量并发评估相关性。串行时延迟与候选数成正比，是原实现最大的性能瓶颈。"""
    docs = state["documents"]
    if not docs:
        return {"documents": []}
    chain = _GRADE_PROMPT | llm | StrOutputParser()
    inputs = [{"question": state["question"], "document": d.page_content} for d in docs]
    try:
        verdicts = chain.batch(inputs, config={"max_concurrency": GRADE_CONCURRENCY})
    except Exception:
        traceback.print_exc()
        return {"documents": docs}  # 评估失败时宁可全部放行，也不要清空上下文
    return {"documents": [d for d, v in zip(docs, verdicts) if v.strip().lower().startswith("yes")]}


def _graph_generate(state: GraphState, llm) -> dict:
    context, _ = build_context(state["documents"])
    history = state.get("history", "")
    history_block = f"\n参考对话历史:\n{history}\n" if history else ""
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "context": context,
            # 用原始问题生成，避免答案去回应被重写过的问法
            "question": state.get("original_question") or state["question"],
            "history_block": history_block,
        }
    )
    return {"generation": answer.strip()}


def _graph_rewrite(state: GraphState, llm) -> dict:
    chain = _REWRITE_PROMPT | llm | StrOutputParser()
    new_q = chain.invoke({"question": state["question"]}).strip()
    return {"question": new_q, "retry_count": state.get("retry_count", 0) + 1}


def _graph_decide(state: GraphState) -> str:
    if state["documents"] or state.get("retry_count", 0) >= MAX_RETRIES:
        return "generate"
    return "rewrite"


def _get_graph(kb_id: str, cfg: RagConfig):
    if kb_id in _graph_cache:
        return _graph_cache[kb_id]

    llm = _get_llm()
    retriever = HybridRetriever(
        kb=get_kb(kb_id), mode=cfg.mode, top_k=cfg.top_k, fetch_k=cfg.fetch_k, llm=llm
    )
    g = StateGraph(GraphState)
    g.add_node("retrieve", lambda s: _graph_retrieve(s, retriever))
    g.add_node("grade", lambda s: _graph_grade(s, _get_scoring_llm()))
    g.add_node("generate", lambda s: _graph_generate(s, llm))
    g.add_node("rewrite", lambda s: _graph_rewrite(s, llm))
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_edge("generate", END)
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("grade", _graph_decide, {"generate": "generate", "rewrite": "rewrite"})

    compiled = g.compile()
    _graph_cache[kb_id] = compiled
    return compiled


def _run_graph(question: str, history: str, cfg: RagConfig) -> QueryResponse:
    t0 = time.perf_counter()
    graph = _get_graph(cfg.kb_id, cfg)
    result = graph.invoke(
        {
            "question": question,
            "original_question": question,
            "history": history,
            "documents": [],
            "generation": "",
            "retry_count": 0,
        }
    )
    _, citations = build_context(result.get("documents", []))
    return QueryResponse(
        answer=result.get("generation", ""),
        sources=_to_source_items(citations),
        session_id="",
        timings_ms={"total": round((time.perf_counter() - t0) * 1000, 1)},
        config=f"graph({cfg.mode},k={cfg.top_k})",
    )


# ===== 路由 =====
@app.get("/health", response_model=HealthResponse)
def health():
    from config import get_embeddings

    llm_ok = embedding_ok = False
    try:
        _get_llm().invoke("ping")
        llm_ok = True
    except Exception:
        pass
    try:
        get_embeddings().embed_query("健康检查")
        embedding_ok = True
    except Exception:
        pass
    return HealthResponse(llm_ok=llm_ok, embedding_ok=embedding_ok, kbs=list_knowledge_bases())


@app.post("/kb")
def create_kb(req: CreateKbRequest, user: str = Depends(_auth)):
    kb_id = req.kb_id or f"kb_{int(time.time())}"
    get_kb(kb_id)
    register_kb(kb_id, user)
    return {"kb_id": kb_id, "owner": user, "message": "知识库已创建"}


@app.get("/kb")
def get_kbs(user: str = Depends(_auth)):
    all_kbs = list_knowledge_bases()
    if not SERVICE_API_KEY:
        return {"kbs": all_kbs}
    return {"kbs": [k for k in all_kbs if is_owner(k, user)]}


@app.get("/kb/{kb_id}/stats")
def kb_stats(kb_id: str, user: str = Depends(_auth)):
    if SERVICE_API_KEY and not is_owner(kb_id, user):
        raise HTTPException(status_code=403, detail="无权访问该知识库")
    return get_kb(kb_id).stats()


@app.get("/kb/{kb_id}/sources")
def kb_sources(kb_id: str, user: str = Depends(_auth)):
    if SERVICE_API_KEY and not is_owner(kb_id, user):
        raise HTTPException(status_code=403, detail="无权访问该知识库")
    return {"kb_id": kb_id, "sources": get_kb(kb_id).list_sources()}


@app.delete("/kb/{kb_id}")
def remove_kb(kb_id: str, user: str = Depends(_auth)):
    _require_write(kb_id, user)
    if not delete_knowledge_base(kb_id):
        raise HTTPException(status_code=404, detail="知识库不存在")
    delete_registry(kb_id)
    _invalidate(kb_id)
    return {"kb_id": kb_id, "deleted": True}


@app.delete("/kb/{kb_id}/sources")
def remove_source(kb_id: str, source: str, user: str = Depends(_auth)):
    """按来源删除。文档更新的正确姿势是「先调这里删掉旧版，再 /ingest 新版」。"""
    _require_write(kb_id, user)
    removed = get_kb(kb_id).delete_by_source(source)
    _invalidate(kb_id)
    return {"kb_id": kb_id, "source": source, "removed_chunks": removed}


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    file: UploadFile = File(...),
    kb_id: str = "default",
    strategy: str = ChunkStrategy.MARKDOWN.value,
    user: str = Depends(_auth),
):
    _require_write(kb_id, user)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail=f"仅支持 {', '.join(SUPPORTED_EXTS)}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".txt") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        docs = load_file(tmp_path)
        if not docs:
            raise HTTPException(status_code=400, detail="文档解析为空")

        filename = file.filename or "unknown"
        # source 用原始文件名而非临时路径，否则引用和按来源删除都会失效
        for d in docs:
            d.metadata["source"] = filename
            d.metadata["file_name"] = filename

        stats = get_kb(kb_id).add_documents(docs, strategy=strategy, source_label=filename)
        _invalidate(kb_id)
        return IngestResponse(
            filename=filename,
            kb_id=kb_id,
            raw_docs=stats.raw_docs,
            chunks=stats.chunks,
            added=stats.added,
            skipped=stats.skipped,
            total_docs=stats.total_in_kb,
            strategy=stats.strategy,
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")
    finally:
        os.unlink(tmp_path)


@app.get("/connectors")
def list_connectors():
    return {"connectors": available_connectors()}


@app.post("/ingest/connector")
def ingest_connector(
    connector_type: str,
    kb_id: str = "default",
    mock: Optional[bool] = None,
    user: str = Depends(_auth),
):
    """从外部系统（Confluence / Jira / GitHub）拉取内容并入库。

    凭据来自 .env（见 config.py），未配置时自动进入 MOCK 模式。
    """
    _require_write(kb_id, user)
    try:
        connector = get_connector(connector_type, _connector_config(connector_type, mock))
        docs = connector.fetch(kb_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"连接器拉取失败: {e}")

    if not docs:
        raise HTTPException(status_code=400, detail="连接器未返回任何内容")

    stats = get_kb(kb_id).add_documents(
        docs, strategy=ChunkStrategy.RECURSIVE, source_label=connector_type
    )
    _invalidate(kb_id)
    return {
        "connector": connector_type,
        "kb_id": kb_id,
        "raw_docs": stats.raw_docs,
        "chunks": stats.chunks,
        "added": stats.added,
        "skipped": stats.skipped,
        "total_docs": stats.total_in_kb,
    }


def _connector_config(connector_type: str, mock: Optional[bool] = None) -> dict:
    from config import (
        CONFLUENCE_API_TOKEN, CONFLUENCE_BASE_URL, CONFLUENCE_EMAIL, CONFLUENCE_SPACES,
        GITHUB_BRANCH, GITHUB_REPOS, GITHUB_TOKEN,
        JIRA_API_TOKEN, JIRA_BASE_URL, JIRA_EMAIL, JIRA_JQL,
    )

    if connector_type == "confluence":
        return {
            "base_url": CONFLUENCE_BASE_URL, "email": CONFLUENCE_EMAIL,
            "api_token": CONFLUENCE_API_TOKEN, "spaces": CONFLUENCE_SPACES, "mock": mock,
        }
    if connector_type == "jira":
        return {
            "base_url": JIRA_BASE_URL, "email": JIRA_EMAIL,
            "api_token": JIRA_API_TOKEN, "jql": JIRA_JQL, "mock": mock,
        }
    if connector_type == "github":
        return {
            "token": GITHUB_TOKEN, "repos": GITHUB_REPOS,
            "branch": GITHUB_BRANCH, "mock": mock,
        }
    raise ValueError(f"未知连接器类型: {connector_type}")


@app.post("/search")
def search(req: SearchRequest, user: str = Depends(_auth)):
    """只检索不生成。答案不对时先看这里 —— 分不清是召回问题还是生成问题，调优就是瞎猜。"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询不能为空")
    pipeline = _get_pipeline(_config_from(req, req.kb_id))
    result = pipeline.retriever.search(req.query)
    _, citations = build_context(result.documents)
    return {
        "query": req.query,
        "mode": result.mode,
        "expanded_queries": result.queries,
        "candidates": result.candidates,
        "timings_ms": result.timings_ms,
        "results": [
            {**c, "score": round(s, 6)} for c, s in zip(citations, result.scores)
        ],
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, user: str = Depends(_auth)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = req.session_id or str(uuid.uuid4())
    cfg = _config_from(req, req.kb_id)
    try:
        append_message(session_id, "user", req.question)
        history = history_text(session_id, HISTORY_TURNS)

        if req.engine == "graph":
            res = _run_graph(req.question, history, cfg)
            res.session_id = session_id
        else:
            ans = _get_pipeline(cfg).answer(req.question, history)
            res = QueryResponse(
                answer=ans.answer,
                sources=_to_source_items(ans.citations),
                session_id=session_id,
                abstained=ans.abstained,
                timings_ms=ans.timings_ms,
                config=ans.config_label,
                queries=ans.queries,
            )

        append_message(session_id, "assistant", res.answer)
        return res
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"问答失败: {e}")


@app.post("/query/stream")
def query_stream(req: QueryRequest, user: str = Depends(_auth)):
    """SSE 流式问答。事件顺序：meta（会话+引用）→ 若干 token → done。

    真正的 token 级流式：引用在生成开始前就发出，前端可以先渲染来源，
    用户会明显感觉响应更快。

    所有事件的 data 都是 JSON 字符串。这一点很关键：token 文本里可能含换行，
    直接塞进 `data:` 会把一帧劈成多帧，前端拼出来的文本就乱了。
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = req.session_id or str(uuid.uuid4())
    cfg = _config_from(req, req.kb_id)

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def event_gen():
        try:
            append_message(session_id, "user", req.question)
            history = history_text(session_id, HISTORY_TURNS)
            pipeline = _get_pipeline(cfg)

            collected = []
            for ev in pipeline.stream(req.question, history):
                if ev["type"] == "retrieved":
                    yield sse("meta", {"session_id": session_id, "sources": ev["citations"]})
                elif ev["type"] == "token":
                    collected.append(ev["text"])
                    yield sse("token", {"text": ev["text"]})
                elif ev["type"] == "done":
                    yield sse("done", {
                        "timings_ms": ev["timings_ms"],
                        "abstained": ev["abstained"],
                        "used_citations": ev.get("used_citations", []),
                    })

            append_message(session_id, "assistant", "".join(collected))
        except Exception as e:
            traceback.print_exc()
            yield sse("error", {"detail": str(e)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        # 关掉 nginx 等反代的缓冲，否则流式会被攒成一整块再吐出来
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions/{session_id}")
def get_session(session_id: str, limit: int = 20, user: str = Depends(_auth)):
    return {"session_id": session_id, "messages": get_messages(session_id, limit)}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str, user: str = Depends(_auth)):
    delete_session(session_id)
    return {"session_id": session_id, "deleted": True}
