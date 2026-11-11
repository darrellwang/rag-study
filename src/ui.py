"""
知识库 Web 界面 (Streamlit)

功能：
  · 知识库选择/创建/删除，来源文件列表与按来源删除
  · 检索策略实时可调（模式 / top_k / 融合 / 重排 / 查询改写）——
    这是 UI 最有价值的部分：参数改完立刻能看到检索结果变化
  · 检索调试面板（只召回不生成，用来判断问题出在召回还是生成）
  · 流式问答，边生成边显示，并展示各阶段耗时
  · 文件上传入库（含切分策略选择）、外部连接器拉取

运行（先启动 api.py）:
  source .venv/bin/activate
  streamlit run src/ui.py --server.port 8501
"""
import json
import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("KB_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("KB_API_KEY", "")


def _headers() -> dict:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def api_get(path: str, **params) -> dict:
    r = requests.get(f"{API_URL}{path}", headers=_headers(), params=params or None, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, **kw) -> dict:
    r = requests.post(f"{API_URL}{path}", headers=_headers(), timeout=600, **kw)
    r.raise_for_status()
    return r.json()


def api_delete(path: str, **params) -> dict:
    r = requests.delete(f"{API_URL}{path}", headers=_headers(), params=params or None, timeout=60)
    r.raise_for_status()
    return r.json()


def render_sources(sources: list, key: str) -> None:
    """统一的来源渲染。原实现两处写法不一致，一处会把整个 dict 直接打出来。"""
    if not sources:
        return
    with st.expander(f"📎 引用来源（{len(sources)} 条）", expanded=False):
        for i, s in enumerate(sources, 1):
            if not isinstance(s, dict):
                st.caption(f"[{i}] {s}")
                continue
            idx = s.get("index") or i
            mark = "✓ " if s.get("used") else ""
            label = s.get("source", "未知")
            if s.get("heading_path"):
                label += f" › {s['heading_path']}"
            if s.get("page") is not None:
                label += f" · 第 {s['page']} 页"
            st.markdown(f"{mark}**[{idx}]** `{label}`")
            if s.get("snippet"):
                st.caption(s["snippet"][:300])


st.set_page_config(page_title="本地知识库", page_icon="📚", layout="wide")
st.title("📚 本地知识库问答")

try:
    health = api_get("/health")
except Exception as e:
    st.error(f"无法连接服务: {e}")
    st.caption(f"请确认 {API_URL} 已启动（uvicorn src.api:app）。若设了 KB_API_KEY 需同步配置。")
    st.stop()

# ---- 侧边栏 ----
with st.sidebar:
    st.header("⚙️ 服务状态")
    c1, c2 = st.columns(2)
    c1.metric("LLM", "✅" if health.get("llm_ok") else "❌")
    c2.metric("Embedding", "✅" if health.get("embedding_ok") else "❌")

    st.divider()
    st.header("🗂️ 知识库")
    kbs = health.get("kbs") or ["default"]
    kb_choice = st.selectbox("选择知识库", kbs)

    new_kb = st.text_input("新建知识库 id（字母/数字/-/_）")
    c1, c2 = st.columns(2)
    if c1.button("创建", use_container_width=True):
        try:
            res = api_post("/kb", json={"kb_id": new_kb or None})
            st.success(f"已创建: {res['kb_id']}")
            st.rerun()
        except Exception as e:
            st.error(f"创建失败: {e}")
    if c2.button("删除当前", use_container_width=True):
        try:
            api_delete(f"/kb/{kb_choice}")
            st.success(f"已删除: {kb_choice}")
            st.rerun()
        except Exception as e:
            st.error(f"删除失败: {e}")

    with st.expander("📊 库内统计与来源管理"):
        try:
            stats = api_get(f"/kb/{kb_choice}/stats")
            st.caption(
                f"{stats['chunks']} 个 chunk ｜ 平均 {stats['avg_chunk_chars']} 字符 "
                f"｜ {len(stats['sources'])} 个来源"
            )
            for src, n in stats["sources"].items():
                cols = st.columns([4, 1])
                cols[0].caption(f"{os.path.basename(src)} · {n} 块")
                if cols[1].button("删", key=f"del_{src}"):
                    api_delete(f"/kb/{kb_choice}/sources", source=src)
                    st.rerun()
        except Exception as e:
            st.caption(f"（读取失败: {e}）")

    st.divider()
    st.header("🔍 检索策略")
    # 这些参数直接决定 RAG 质量，放在最显眼的位置以便随时试
    mode = st.selectbox(
        "召回模式", ["hybrid", "vector", "mmr", "bm25"],
        help="hybrid = 向量 + BM25 双通道融合，生产默认；bm25 对专有名词/错误码更稳",
    )
    top_k = st.slider("交给模型的片段数 top_k", 1, 10, 4)
    fetch_k = st.slider("召回候选数 fetch_k", top_k, 50, max(20, top_k))
    fusion = st.radio("融合方式", ["rrf", "weighted"], horizontal=True,
                      help="RRF 只看名次，无需调参；weighted 需要评估集来定权重")
    use_rerank = st.checkbox("LLM 重排（更准，但明显更慢）")
    use_multi_query = st.checkbox("Multi-Query 改写（提升召回）")
    use_hyde = st.checkbox("HyDE 改写（术语密集场景有效）")
    engine = st.radio("问答引擎", ["pipeline", "graph"], horizontal=True,
                      help="pipeline = 混合检索+引用+拒答；graph = LangGraph 自我纠错")

    retrieval_opts = {
        "mode": mode, "k": top_k, "fetch_k": fetch_k, "fusion": fusion,
        "use_rerank": use_rerank, "use_multi_query": use_multi_query, "use_hyde": use_hyde,
    }

    st.divider()
    st.header("📥 上传文档入库")
    strategy = st.selectbox("切分策略", ["markdown", "recursive", "token"],
                            help="markdown 会把标题路径注入片段开头，技术文档首选")
    uploaded = st.file_uploader("选择文件", type=["txt", "md", "markdown", "pdf", "csv", "json", "jsonl"])
    if uploaded is not None and st.button("入库", type="primary", use_container_width=True):
        with st.spinner("切分并向量化中..."):
            try:
                files = {"file": (uploaded.name, uploaded.getbuffer(), uploaded.type or "text/plain")}
                res = api_post(f"/ingest?kb_id={kb_choice}&strategy={strategy}", files=files)
                st.success(
                    f"✅ {res['filename']}：新增 {res['added']} 块"
                    + (f"，跳过重复 {res['skipped']} 块" if res["skipped"] else "")
                    + f"，库内共 {res['total_docs']} 块"
                )
            except Exception as e:
                st.error(f"入库失败: {e}")

    st.divider()
    st.header("🔌 连接外部系统")
    try:
        conns = api_get("/connectors").get("connectors", [])
    except Exception:
        conns = []
    if conns:
        conn_type = st.selectbox("选择系统", conns)
        use_mock = st.checkbox("使用 MOCK 占位数据", value=True)
        if st.button("拉取并入库", use_container_width=True):
            with st.spinner(f"从 {conn_type} 拉取中..."):
                try:
                    res = api_post(
                        f"/ingest/connector?connector_type={conn_type}&kb_id={kb_choice}"
                        + (f"&mock={'true' if use_mock else 'false'}")
                    )
                    st.success(f"✅ 拉取 {res['raw_docs']} 篇，新增 {res['added']} 块")
                except Exception as e:
                    st.error(f"连接器入库失败: {e}")
    else:
        st.caption("（服务未返回可用连接器）")


# ---- 主区 ----
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

chat_tab, debug_tab = st.tabs(["💬 问答", "🔬 检索调试"])

with debug_tab:
    st.caption("只做召回不生成。答案不对时先来这里看检索到了什么 —— "
               "分不清是召回问题还是生成问题，调优就只能靠猜。")
    debug_q = st.text_input("检索查询", key="debug_query")
    if st.button("检索", key="do_search") and debug_q.strip():
        try:
            res = api_post("/search", json={"query": debug_q, "kb_id": kb_choice, **retrieval_opts})
            cols = st.columns(3)
            cols[0].metric("候选数", res["candidates"])
            cols[1].metric("模式", res["mode"])
            cols[2].metric("总耗时", f"{res['timings_ms'].get('total', 0)} ms")
            if len(res["expanded_queries"]) > 1:
                st.caption("改写后的检索查询：" + " ｜ ".join(res["expanded_queries"]))
            st.json(res["timings_ms"])
            for item in res["results"]:
                st.markdown(f"**[{item['index']}]** `{item['source']}` — score `{item['score']}`")
                st.caption(item["snippet"])
                st.divider()
        except Exception as e:
            st.error(f"检索失败: {e}")

with chat_tab:
    c1, c2 = st.columns([5, 1])
    c1.caption(f"知识库 `{kb_choice}` ｜ 会话 `{st.session_state.session_id[:8]}` ｜ 策略 `{mode}` k={top_k}")
    if c2.button("清空会话"):
        try:
            api_delete(f"/sessions/{st.session_state.session_id}")
        except Exception:
            pass
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            render_sources(m.get("sources", []), key=f"hist_{i}")
            if m.get("timings"):
                st.caption(f"耗时 {m['timings']}")

    if prompt := st.chat_input("问我关于知识库的任何问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            payload = {
                "question": prompt,
                "kb_id": kb_choice,
                "session_id": st.session_state.session_id,
                "engine": engine,
                **retrieval_opts,
            }
            sources: list = []
            timings: dict = {}
            answer = ""

            if engine == "graph":
                # 纠错图会循环重试，中间没有可流式的稳定输出，走同步接口
                with st.spinner("检索 + 自我纠错中..."):
                    try:
                        res = api_post("/query", json=payload)
                        answer, sources = res.get("answer", ""), res.get("sources", [])
                        timings = res.get("timings_ms", {})
                        st.session_state.session_id = res.get("session_id", st.session_state.session_id)
                    except Exception as e:
                        answer = f"❌ 问答失败: {e}"
                st.markdown(answer)
            else:
                # 流式：先出来源，再逐字出答案
                placeholder = st.empty()
                source_slot = st.container()
                try:
                    with requests.post(
                        f"{API_URL}/query/stream", headers=_headers(), json=payload,
                        stream=True, timeout=600,
                    ) as resp:
                        resp.raise_for_status()
                        event = None
                        for line in resp.iter_lines(decode_unicode=True):
                            if not line:
                                continue
                            if line.startswith("event: "):
                                event = line[7:].strip()
                                continue
                            if not line.startswith("data: "):
                                continue
                            data = json.loads(line[6:])
                            if event == "meta":
                                sources = data.get("sources", [])
                                st.session_state.session_id = data.get(
                                    "session_id", st.session_state.session_id
                                )
                                with source_slot:
                                    render_sources(sources, key="live")
                            elif event == "token":
                                answer += data.get("text", "")
                                placeholder.markdown(answer + "▌")
                            elif event == "done":
                                timings = data.get("timings_ms", {})
                                used = set(data.get("used_citations", []))
                                for s_ in sources:
                                    s_["used"] = s_.get("index") in used
                            elif event == "error":
                                answer += f"\n\n❌ {data.get('detail')}"
                            event = None
                    placeholder.markdown(answer)
                except Exception as e:
                    answer = f"❌ 问答失败: {e}"
                    placeholder.markdown(answer)

            if timings:
                st.caption(f"耗时 {timings}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources, "timings": timings}
        )
