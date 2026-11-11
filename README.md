# RAG 学习环境 (LangChain + LangGraph + 本地大模型)

在 WSL 中使用 LangChain / LangGraph / RAG，连接 Windows 上由 **LMStudio** 提供的本地大模型。

## 架构概览

```
Windows                          WSL (Linux)
┌─────────────────────┐         ┌──────────────────────────────┐
│   LMStudio          │  HTTP   │  Python 脚本                  │
│  (OpenAI 兼容接口)  │◄────────┤  LangChain + LangGraph + RAG  │
│  localhost:1234/v1  │         │  Chroma 向量库                │
└─────────────────────┘         └──────────────────────────────┘
```

LMStudio 暴露 OpenAI 兼容接口，因此直接用 `langchain-openai` 的 `ChatOpenAI` / `OpenAIEmbeddings`，
只需把 `base_url` 指向 `http://localhost:1234/v1`。

## 前置准备

### 1. Windows 端：启动 LMStudio

1. 打开 LMStudio，在 **Developer**（开发者）标签页启动 Local Server（默认端口 `1234`）。
2. 加载：
   - 一个 **对话模型**（如 `qwen/qwen3.6-35b-a3b`）
   - 一个 **Embedding 模型**（如 `text-embedding-embeddinggemma-300m`）
3. 确认 Server 状态为 `Running`。

> 在 WSL 镜像网络模式下，WSL 可直接通过 `localhost:1234` 访问 Windows 上的 LMStudio。

### 2. WSL 端：安装依赖

```bash
cd ~/workspace/rag-study
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置模型

编辑 `.env`，选择 LMStudio 中已加载的模型：

```env
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio
LLM_MODEL=qwen/qwen3.6-35b-a3b
EMBEDDING_MODEL=text-embedding-embeddinggemma-300m
```

> 可用模型列表见 LMStudio，或在 WSL 中运行：
> `curl -s http://localhost:1234/v1/models`

## 使用方法

每次使用前先激活虚拟环境：

```bash
source .venv/bin/activate
```

然后按顺序运行示例（循序渐进）：

| 脚本 | 说明 | 命令 |
|------|------|------|
| `01_test_connection.py` | 测试 LMStudio 连接（对话 + embedding） | `python src/01_test_connection.py` |
| `02_basic_chat.py` | 基础多轮对话（带历史记忆） | `python src/02_basic_chat.py` |
| `03_basic_rag.py` | 基础 RAG（检索 + 生成） | `python src/03_basic_rag.py` |
| `04_rag_langgraph.py` | 进阶 RAG（LangGraph 自我纠错） | `python src/04_rag_langgraph.py` |
| `05_tool_use.py` | LangChain 工具调用（Function Calling） | `python src/05_tool_use.py` |
| `06_multi_agent.py` | LangGraph 多智能体协作（研究→写作→审校） | `python src/06_multi_agent.py` |
| `07_map_reduce_summarize.py` | Map-Reduce 长文档摘要 | `python src/07_map_reduce_summarize.py` |
| `08_workflow_router.py` | LangGraph 条件路由（按问题类型分发） | `python src/08_workflow_router.py` |

交互式脚本中输入 `quit` 退出。

> **注意**: `06` 和 `08` 涉及多轮 LLM 调用，本地大模型每次调用约 30-60 秒，请耐心等待。

## 项目结构

```
rag-study/
├── .env                  # 模型与服务配置
├── .env.example          # 配置模板
├── requirements.txt      # Python 依赖
├── data/                 # 知识库文档（txt/md/pdf/csv/json），含 RAG 知识示例语料
├── eval/                 # 评估数据集（问题 + 标准来源 + 关键词）
├── tests/                # 离线单元测试（用假 embedding，无需 LMStudio）
├── chroma_db/            # Chroma 向量库持久化数据（自动生成）
├── Dockerfile           # 知识库服务镜像（API + UI 层）
├── docker-compose.yml   # 容器编排（api + ui 两服务）
├── .dockerignore        # 构建上下文排除项
└── src/
    ├── config.py         # 集中配置：LMStudio 连接 + 切分/检索参数
    ├── rag_utils.py      # 兼容层，转调 rag/ 包（保持 03/04 脚本可用）
    ├── api.py            # FastAPI 服务（多知识库/会话/鉴权/检索调试/真流式）
    ├── ui.py             # Streamlit 界面（检索策略面板 + 流式问答 + 调试页）
    ├── rag/              # ★ RAG 核心包
    │   ├── loaders.py        # 多格式加载（txt/md/pdf/csv/json）
    │   ├── chunking.py       # 三种切分策略（递归/token/Markdown 结构感知）
    │   ├── store.py          # 本地知识库：幂等入库、按来源增删、统计
    │   ├── bm25.py           # BM25 稀疏检索（纯 Python 实现，含公式推导注释）
    │   ├── fusion.py         # RRF / 加权分数融合
    │   ├── rerank.py         # LLM 重排 + 可选 cross-encoder
    │   ├── query_transform.py# Multi-Query / HyDE / 问题分解
    │   ├── retriever.py      # 混合检索器（召回→融合→重排 全可配）
    │   ├── pipeline.py       # 端到端管道（带引用生成、拒答、耗时分解）
    │   └── evaluate.py       # 评估：Hit@k / Recall@k / MRR / nDCG + LLM-as-judge
    ├── 01_test_connection.py   # 连接测试
    ├── 02_basic_chat.py        # 基础多轮对话
    ├── 03_basic_rag.py         # 基础 RAG
    ├── 04_rag_langgraph.py     # LangGraph 纠错 RAG
    ├── 05_tool_use.py          # 工具调用 (Function Calling)
    ├── 06_multi_agent.py       # 多智能体协作
    ├── 07_map_reduce_summarize.py  # Map-Reduce 长文档摘要
    ├── 08_workflow_router.py   # 条件路由工作流
    ├── 09_advanced_retrieval.py# ★ 六种召回策略同题对比
    ├── 10_query_transform.py   # ★ Multi-Query / HyDE / 问题分解
    ├── 11_rag_eval.py          # ★ 离线评估与 A/B 对比
    └── 12_kb_manage.py         # ★ 本地知识库全生命周期 CLI
```

## 添加自己的知识库

支持 `.txt / .md / .pdf / .csv / .json / .jsonl`。**不再需要删库重建**——
chunk id 由内容哈希决定，重复入库会自动跳过：

```bash
cp /path/to/your.pdf data/
python src/12_kb_manage.py add data/your.pdf          # 单文件
python src/12_kb_manage.py add data/ --strategy markdown  # 整个目录
python src/12_kb_manage.py stats                      # 看看入了多少
```

文档改动后用 `update`（先按来源删除旧 chunk 再插入），而不是 `add`：

```bash
python src/12_kb_manage.py update data/your.md
python src/12_kb_manage.py rm-source your.md     # 下架某个文档
```

## 关键实现说明（踩坑记录）

1. **代理冲突**：WSL 常配置 `ALL_PROXY=socks5://...` 用于科学上网，httpx 会尝试经代理访问
   localhost 的 LMStudio 而报错。`config.py` 中为本地连接传入了 `trust_env=False` 的 httpx 客户端来绕过。
2. **Embedding 输入格式**：`OpenAIEmbeddings` 默认 `check_embedding_ctx_length=True` 会把文本转成
   token ID 数组发送，但 LMStudio 只接受字符串，会报 400。已在 `config.py` 中关闭该选项。
3. **超时**：本地大模型推理较慢（尤其 70B），HTTP 超时设为 300 秒。

## LangGraph RAG 工作流（04 脚本）

```
START → retrieve(检索) → grade(评估相关性) ──┐
                              相关≥1篇?      │
                        ┌───────────────────┤
                       是│                  否│
                        ▼                   ▼
                   generate            rewrite(重写问题)
                        │                   │
                        ▼                   └─→ retrieve(重试, 最多2次)
                       END
```

检索到的文档全不相关时，LangGraph 让模型重写问题后重新检索，而非硬编答案，从而减少幻觉。
这正是用图结构编排「循环 + 条件分支」的价值。

## 阶段 1：知识库服务（FastAPI）

把 `03/04` 的 RAG 能力封装为 HTTP 服务，可作为独立知识库后端，供前端或任意客户端调用。
代码见 `src/api.py`，复用 `config.py` 与 `rag_utils.py`。

### 安装服务依赖

```bash
source .venv/bin/activate
pip install -r requirements.txt   # 已加入 fastapi / uvicorn / python-multipart
```

### 启动服务

```bash
source .venv/bin/activate
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

> 仅绑定 `127.0.0.1`，不暴露公网。需远程访问时走 WSL 端口转发即可。
> 本地 35B 模型推理慢，`--reload` 仅开发用，生产可去掉。

### API 说明

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 检查 LMStudio（LLM/embedding）连通性与向量库文档数 |
| `POST` | `/ingest` | 上传 `.txt` / `.pdf` 文档，切分后入库，返回块数与总量 |
| `POST` | `/query` | 问题 → 检索 → LangGraph 纠错生成，返回答案与来源文本 |
| `POST` | `/query/stream` | 同上，但 SSE 流式逐段返回，避免前端卡死 |

### 快速验证

```bash
# 健康检查
curl -s http://127.0.0.1:8000/health

# 上传文档入库
curl -s -F "file=@data/example.txt" http://127.0.0.1:8000/ingest

# 问答
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 RAG？"}'
```

问答接口复用 `04` 的纠错逻辑：检索到的文档全部不相关时，模型会重写问题重新检索（最多 2 次），
而非硬编答案，从而减少本地小模型幻觉。

## 阶段 2：Web 界面（Streamlit）

聊天式前端，调用阶段 1 的 API，无需命令行即可问答、上传文档、查看来源引用。

### 安装与启动

```bash
source .venv/bin/activate
pip install -r requirements.txt   # 已加入 streamlit / requests
streamlit run src/ui.py --server.port 8501
```

浏览器打开 http://127.0.0.1:8501 ：

- **侧边栏**：实时显示服务健康（LLM / Embedding / 文档数），可上传 `.txt`/`.pdf` 一键入库
- **主聊天区**：输入问题，回答下方可展开「来源」查看检索到的原文片段

> 界面默认连 `http://127.0.0.1:8000`，可用环境变量 `KB_API_URL` 覆盖（如指向容器服务）。
> 需先启动 `uvicorn src.api:app`（阶段 1）；本地模型慢，问答时显示「检索+生成中」属正常。

## 阶段 3：容器化部署（Docker）

把 API + UI 封装为容器，推理仍由 **Windows 上的 LMStudio** 完成（仅服务层容器化）。

### 架构要点

```
Windows LMStudio (localhost:1234)
        ▲  host.docker.internal (宿主网关)
        │
┌───────┴─────────────────────── WSL / Docker ───────────────┐
│  kb-api  (容器, :8000)   ◄──►   kb-ui  (容器, :8501)        │
│  挂载卷: chroma_data / doc_data                              │
└────────────────────────────────────────────────────────────┘
```

容器内的 `localhost` 指向容器自身，因此访问宿主 LMStudio 必须用 `host.docker.internal`
（compose 已通过 `extra_hosts: host.docker.internal:host-gateway` 注入）。

### 启动

```bash
# 1. 修改 .env，让容器能访问宿主 LMStudio：
#    LMSTUDIO_BASE_URL=http://host.docker.internal:1234/v1
# 2. 构建并启动
docker compose up --build -d
```

- API：http://127.0.0.1:8000
- UI ：http://127.0.0.1:8501

### 数据持久化

`chroma_db` 与 `data` 通过命名卷（`chroma_data` / `doc_data`）持久化，重建容器不丢数据。
停止：`docker compose down`（卷保留）；彻底清理：`docker compose down -v`。

> 仅绑定本地端口，不暴露公网。如需远程访问，请在宿主机/路由器做端口转发而非改绑 `0.0.0.0` 到外网。

## 阶段 4：生产化增强

在阶段 1~3 基础上补齐三项可用性能力，接口全部向后兼容（旧调用不加参数即走 `default` 库、无会话）。

### 4.1 多知识库隔离

不同主题/用户的数据分库存储，互不污染：

- `POST /kb` 创建知识库（可指定 `kb_id`，留空自动生成）— 需 API Key（见 4.3）
- `GET /kb` 列出所有知识库
- `DELETE /kb/{kb_id}` 删除知识库（含向量数据）
- 入库/问答时带 `kb_id` 参数路由到对应库；每个库是独立的 Chroma 目录 `<CHROMA_PERSIST_DIR>/<kb_id>`
- `kb_id` 仅允许字母/数字/`-`/`_`（防路径穿越）

```bash
curl -s -X POST http://127.0.0.1:8000/kb -H "X-API-Key: $KB_API_KEY" -d '{"kb_id":"hr"}'
curl -s -F "file=@hr.pdf" "http://127.0.0.1:8000/ingest?kb_id=hr"
curl -s -X POST http://127.0.0.1:8000/query -d '{"question":"年假几天？","kb_id":"hr"}'
```

### 4.2 会话历史（多轮对话）

- 请求带 `session_id` 即维护该会话的多轮上下文；不传则服务端自动生成并返回
- 历史最近 6 轮会拼入 RAG prompt 的「对话历史」段，使追问能理解上下文（如「它是什么时候发布的？」）
- `GET /sessions/{session_id}` 可查看某会话记录
- 注意：历史为进程内存存储，重启服务即丢失；生产可换 Redis/SQLite（见下方扩展点）

```bash
SID=$(curl -s -X POST http://127.0.0.1:8000/query -d '{"question":"LangChain 是什么？"}' | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")
curl -s -X POST http://127.0.0.1:8000/query -d "{\"question\":\"它和 LangGraph 什么关系？\",\"session_id\":\"$SID\"}"
```

### 4.3 轻量 API Key 鉴权（可选）

- `.env` 设 `KB_API_KEY=xxx` 后，写操作（`/kb` `/ingest` `/query` `/query/stream`）需在 Header 带 `X-API-Key`
- 留空则不校验（本地默认关闭，方便调试）
- 前端 `src/ui.py` 读取环境变量 `KB_API_KEY` 自动附加到请求

### 更新后的 API 总览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET`  | `/health` | 健康检查 + 知识库列表 |
| `POST` | `/kb` | 创建知识库（需 Key） |
| `GET`  | `/kb` | 列出知识库 |
| `DELETE`| `/kb/{kb_id}` | 删除知识库（需 Key） |
| `POST` | `/ingest` | 上传文档入库（支持 `kb_id`，需 Key） |
| `POST` | `/query` | 问答（支持 `kb_id`/`session_id`） |
| `POST` | `/query/stream` | 流式问答（同上，含 meta 事件返回 session_id/来源） |
| `GET`  | `/sessions/{sid}` | 查看会话历史 |

### 扩展点（已实现）

以下三项原列为扩展点，现均已落地（v0.5.0）：

1. **会话历史持久化（SQLite）**
   - `src/session_store.py`：用 SQLite 存储每轮消息（库文件 `db/sessions.db`）
   - 重启服务不丢会话；`append_message` / `get_messages` / `history_text` / `delete_session`
   - 最近 6 轮仍注入 RAG prompt 的「对话历史」段

2. **用户级权限（kb owner）**
   - `src/kb_registry.py`：为每个知识库登记 owner（库文件 `db/kb_registry.db`）
   - owner 由请求携带的 `X-API-Key` 派生（`u_<sha256 前8位>`）；未启用鉴权时为 `anonymous`，不校验
   - 创建（`POST /kb`）登记 owner；删除（`DELETE /kb/{kb_id}`）、入库（`POST /ingest`）校验归属；启用鉴权后 `GET /kb` 只返回自己拥有的库
   - 注：当前为单用户级（一个 Key = 一个用户），未做多用户注册；如需多用户可在 `_auth` 处扩展

3. **文档来源可追溯**
   - 入库时 `rag_utils.tag_documents` 为每个 chunk 注入 `source`（文件名）/ `kb_id` metadata
   - 回答时 `rag_utils.sources_from_docs` 抽取**结构化来源**（去重：`source` + `kb_id` + 片段），前端展示文件名而非整段原文
   - `QueryResponse.sources` 为 `List[SourceItem]`，`/query/stream` 的 meta 事件同样返回结构化来源

> 容器部署时 `db_data` 卷已挂载，SQLite 库随容器持久化。

## 扩展 2：外部系统连接器（Confluence / Jira / GitHub）

把在线系统的内容拉取成知识库文档，复用同一套切分 / 向量化 / 检索链路。

### 架构

```
Connector (拉取) ──► LangChain Document (带 source metadata)
       │
       ▼
RecursiveCharacterTextSplitter 切分
       │
       ▼
tag_documents 注入 source / source_type / source_id
       │
       ▼
Chroma 入库（按 kb_id 路由）
```

- `src/connectors/__init__.py`：抽象基类 `BaseConnector` + 工厂 `get_connector(type, config)`
- `src/connectors/confluence.py` / `jira.py` / `github.py`：三个内置连接器
- 每个连接器拉取后统一写入 `source_type`（confluence/jira/github）、`source_id`（去重主键）、
  `source`（可点击链接），使回答能精确溯源到某篇 wiki 页 / 某个 issue / 某个 PR

### MOCK 模式与真实凭据

未配置凭据时自动 MOCK（返回样例数据，便于先把整条链路跑通）；在 `.env` 填好凭据后切换真实 API。
各连接器凭据见 `.env.example`（`CONFLUENCE_*` / `JIRA_*` / `GITHUB_*`）。

> 外部系统走公司内网 / 公网，**不**绕过系统代理（与 LMStudio 的 localhost 相反），
> 因此连接器内部使用 `httpx.Client()` 默认信任 `*_PROXY` 环境变量。若 WSL 已配 `ALL_PROXY` 即可直连。

### API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET`  | `/connectors` | 列出可用连接器类型 |
| `POST` | `/ingest/connector` | 拉取某系统内容并入库（参数 `connector_type` / `kb_id` / `mock`） |

```bash
# 查看可用连接器
curl -s http://127.0.0.1:8000/connectors

# 用 MOCK 数据快速验证
curl -s -X POST "http://127.0.0.1:8000/ingest/connector?connector_type=confluence&kb_id=demo&mock=true"

# 配置好凭据后拉取真实内容（mock 参数省略，按凭据自动判断）
curl -s -X POST "http://127.0.0.1:8000/ingest/connector?connector_type=jira&kb_id=demo"
```

### UI

Streamlit 侧边栏「🔌 连接外部系统」：下拉选系统 → 勾选「使用 MOCK」→ 点「拉取并入库」。
入库后切到对应知识库即可对话，来源会显示 `confluence:...` / `jira:...` / `github:...` 链接。

### 新增连接器

1. 在 `src/connectors/` 新建 `xxx.py`，继承 `BaseConnector`，实现 `_fetch_raw()` 返回 `Document` 列表
2. 在 `src/connectors/__init__.py` 末尾 `register("xxx", XxxConnector)`
3. （可选）在 `src/config.py` / `.env.example` 增加对应凭据配置

## 阶段 5：进阶检索与评估（RAG 效果优化）

前面几个阶段把「链路」跑通了，但检索层一直是最朴素的单路向量 top-k。
这一阶段是 RAG 从 demo 走向可用的分水岭。

### 5.1 核心认知

**RAG 的效果上限由检索决定。** 答案里不可能出现没被检索到的信息。
所以效果不好时排查顺序是固定的：

```
先看检索层指标（Hit@k / MRR）
    │
    ├─ 差 ──► 改切分粒度 / 加 BM25 通道 / 加重排。不要碰提示词
    │
    └─ 好 ──► 再看生成层（忠实度 / 相关度）→ 这时才是提示词或模型的问题
```

### 5.2 检索流水线

```
question
   │
   ├─(可选) Multi-Query / HyDE 改写 → 多条检索 query
   │
   ├─ 向量通道（语义近似强，精确串弱）  ┐
   ├─ BM25  通道（专有名词/错误码兜底）┘ 各取 fetch_k=20
   │
   ├─ RRF 融合（只看名次，跨通道量纲无关）
   │
   ├─(可选) LLM 重排 + 相关性门槛
   │
   └─ top_k=4 → 带编号上下文 → 生成（要求标注 [1][2]）→ 或拒答
```

每一步都可以单独开关，配合 `11_rag_eval.py` 做 A/B。
向量与 BM25 为什么必须一起用：前者匹配「登录转圈 ↔ 认证服务超时」，
后者匹配 `check_embedding_ctx_length` 这种 embedding 模型没见过的内部标识。

### 5.3 上手顺序

```bash
# 1. 看六种召回策略在同一问题上的差异（换不同类型的问题多试几次）
python src/09_advanced_retrieval.py "chunk_size 设多大合适"
python src/09_advanced_retrieval.py "check_embedding_ctx_length 是什么"  # 看 BM25 如何反超

# 2. 看查询改写做了什么
python src/10_query_transform.py

# 3. 量化效果：先跑只评检索的快模式
python src/11_rag_eval.py --retrieval-only --all

# 4. 端到端评估（慢，含 LLM-as-judge）
python src/11_rag_eval.py --full --config hybrid --report eval/report.json

# 5. 知识库管理与调试
python src/12_kb_manage.py ls
python src/12_kb_manage.py search "BM25 参数" --mode hybrid
python src/12_kb_manage.py chunk-compare data/rag_retrieval.md
```

### 5.4 可调参数（`.env` 或环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CHUNK_SIZE` | 500 | 切分粒度（字符）。中文常见区间 300~800 |
| `CHUNK_OVERLAP` | 80 | 重叠，一般取 chunk_size 的 10%~20% |
| `TOP_K` | 4 | 最终交给模型的片段数 |
| `FETCH_K` | 20 | 召回候选数，留给融合和重排筛选的空间 |
| `RETRIEVAL_MODE` | hybrid | `vector` / `mmr` / `bm25` / `hybrid` |
| `VECTOR_WEIGHT` | 0.6 | 加权融合时向量通道权重（RRF 模式下不用） |
| `RRF_K` | 60 | RRF 平滑常数 |
| `USE_RERANK` | false | 是否默认启用 LLM 重排（本地模型下很慢） |

### 5.5 评估指标怎么读

| 现象 | 结论 | 该做什么 |
| --- | --- | --- |
| `hit@k` 低 | 召回阶段就没捞到 | 改切分策略、加 BM25 通道、加大 fetch_k |
| `hit@k` 高但 `mrr` 低 | 捞到了但排得靠后 | 加重排，收益最直接 |
| 检索层都好但 `faithfulness` 低 | 模型没老实用上下文 | 这时才该调提示词或换模型 |
| `abstain_rate` 过低 | 库外问题也硬答 | 幻觉风险高，提高相关性门槛 |
| `ndcg` 涨了但延迟翻倍 | 权衡问题 | 看业务能接受多少延迟 |

评估集在 `eval/qa_dataset.json`，最后一条是故意放的库外问题（太阳系行星），
用来检验拒答能力。加自己的样例时照抄这个格式即可。

> LLM-as-judge 的评判模型最好强于被测模型。本地只有单一模型时，
> 「自己给自己打分」会系统性偏高，应以检索层指标和关键词覆盖率为主。

### 5.6 新增接口

| 接口 | 说明 |
| --- | --- |
| `POST /search` | **只检索不生成**。答案不对时的第一站，能立刻分清是召回问题还是生成问题 |
| `GET /kb/{id}/stats` | chunk 数、来源分布、平均长度 |
| `GET /kb/{id}/sources` | 来源文件列表及各自 chunk 数 |
| `DELETE /kb/{id}/sources?source=x.md` | 按来源删除，文档更新/下架用 |
| `DELETE /sessions/{sid}` | 清除会话历史 |

`/query` 与 `/query/stream` 新增可选参数：`mode`、`k`、`fetch_k`、`fusion`、
`use_rerank`、`use_multi_query`、`use_hyde`、`engine`（`pipeline` / `graph`）。
全部有默认值，老客户端不传照常工作。

```bash
# 只检索，看召回了什么
curl -s -X POST localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"BM25 的 k1 参数","kb_id":"default","mode":"hybrid","k":4}' | jq

# 问答（返回引用清单 + 各阶段耗时）
curl -s -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"chunk 怎么切","kb_id":"default","mode":"hybrid","use_rerank":false}' | jq
```

### 5.7 ★ 最重要的一条：别用推理模型做打分

重排和文档相关性评估本质是**分类任务**，期望输出只有一个数字或 yes/no。
但 Qwen3 等主流本地模型默认开思维链，为了吐出一个 `10`，会先生成几百到几千个
隐藏的 reasoning token。同一次打分实测（`qwen/qwen3.6-35b-a3b`）：

| 配置 | 耗时 | output tokens | 其中 reasoning | 结果 |
| --- | --- | --- | --- | --- |
| 默认（开思考） | 48.2s | 2543 | 2539 | `5` |
| `extra_body` 关 thinking | 41.2s | 2176 | 2172 | `5`（LMStudio 不认这个参数） |
| **`reasoning_effort="none"`** | **4.9s** | **2** | **0** | `5` |
| `max_tokens=8` | 1.4s | 8 | 8 | `''`（被截断，不可用） |

**差一个数量级，而打分结果完全一致。** 重排要跑 `fetch_k` 次，这个差异直接决定
「重排能不能用」——实测端到端从 163 秒降到 33 秒。

所以 `config.py` 提供了独立的 `get_scoring_llm()`，重排、相关性评估、
LLM-as-judge 全部走它；生成仍用开思考的模型（答案质量需要推理）。
它会一次性探测服务端是否支持该参数，不支持就安静降级。

```python
from config import get_llm, get_scoring_llm

llm = get_llm(temperature=0)      # 生成：保留思维链
scorer = get_scoring_llm()        # 打分/分类：关闭思维链
```

### 5.8 这一阶段修掉的问题

| 问题 | 后果 | 修法 |
| --- | --- | --- |
| 重复上传产生重复 chunk | top-k 被同一段内容占满，浪费上下文预算 | chunk id = `sha1(kb_id+source+正文)`，入库前查重 |
| 无法更新/删除文档 | 改一个字就得重建整库 | metadata 记 source，提供按来源删除 |
| `list_kbs()` 把 Chroma 段目录当知识库 | UI 里冒出 UUID 命名的假知识库 | 只认含 `chroma.sqlite3` 的子目录 |
| 删库后重建同名知识库报 readonly | chromadb 全局缓存了失效的 sqlite 句柄 | 删除前 `clear_system_cache()` |
| `/query` 每次重建 LangGraph | 只为注入对话历史，白白浪费编译开销 | 历史放进图的 state，图按 kb 缓存 |
| 文档相关性评估逐条串行调 LLM | 延迟与候选数成正比 | 改批量并发（`max_concurrency=4`） |
| `/query/stream` 是假流式 | 先算完整答案再切片，首字延迟等于全量延迟 | 改真 token 流，引用先于答案发出 |
| SSE 直接把 token 塞进 `data:` | token 含换行会把一帧劈成多帧 | 所有事件用 JSON 载荷 |
| 用 `vs._collection.count()` 私有 API | 版本升级易碎 | 收口到 `KnowledgeBase.count()` |
| 混合检索 top-k 里出现**重复片段** | 向量通道的 Document 没有 `_id`、BM25 通道有，同一 chunk 被算成两个 → 摧毁了混合检索的意义 | 去重 key 只用「来源 + 正文」哈希 |
| 用推理模型做打分 | 重排慢一个数量级（163s vs 33s） | 独立的 `get_scoring_llm()`，见 5.7 |
| 引用清单列出全部召回结果 | 只用了 1 条却展示 4 条「依据」，用户点开发现无关就不再信引用 | 解析答案里的 `[n]` 标记实际引用，流式在 `done` 事件补发 |
| 拒答检测误伤正常答案 | 问「为什么要有拒答机制」时答案本身含「无法回答」，被误判成拒答，`abstain_rate` 指标算废 | 关键词 + 长度 ≤ 60 字 + 无引用标记，三条同时满足才算拒答 |
| `RagConfig.label()` 不含融合方式 | A/B 表格出现两行同名 `hybrid(k=4)`，没法比 | label 带上 fusion 与各开关 |
| 评估脚本被管道块缓冲 | 长任务看不到进度，像卡死 | `sys.stdout.reconfigure(line_buffering=True)` |
| `--retrieval-only` 仍会调 LLM | rerank 在检索阶段，16×20=320 次推理，跑了 15 分钟没结束 | 拆成 `CHEAP_PRESETS` / `LLM_PRESETS`，后者需显式 `--include-llm` |
| 自定义 Chroma collection 名 | 已有知识库「库还在但内容没了」——Chroma 会在同目录静默新建空 collection，不报错 | 沿用 langchain-chroma 默认名 `langchain`，并加测试锁死 |

> **历史数据提醒**：如果你的 `chroma_db/` 下有 UUID 命名的目录（如
> `94f920a5-.../`），那是上面第三个 bug 留下的痕迹——它原本是根库的段目录，
> 被误列成知识库后被写入了真实数据。确认里面没有需要的内容后可以直接删掉。
> 默认知识库现在统一存放在 `chroma_db/default/`。

### 5.9 实测基线（`qwen3.6-35b-a3b` + `embeddinggemma-300m`）

在 `data/` 三份 RAG 语料（16 chunk）+ `eval/qa_dataset.json`（16 题）上：

**纯检索 A/B**（`--retrieval-only --all`，3.5 秒跑完）

| 配置 | hit@k | mrr | ndcg@k | 延迟 |
| --- | --- | --- | --- | --- |
| vector(k=4) | 0.938 | 0.938 | 0.891 | 15.8 ms |
| bm25(k=4) | 0.938 | 0.906 | 0.885 | **0.3 ms** |
| mmr(k=4) | 0.938 | 0.938 | 0.891 | 17.6 ms |
| **hybrid(k=4)** | 0.938 | 0.938 | **0.926** | 15.4 ms |

**端到端**（`--full --config hybrid`，16 题约 3.5 分钟）

`hit@k=0.938`｜`mrr=0.938`｜`faithfulness=5.0`｜`relevance=4.71`｜
`keyword_coverage=0.867`｜`abstain_rate=0.0625`｜平均 5.8 秒/题

**流式延迟**：来源 0.04s 送达 → 首字 5.0s → 完成 6.9s（116 token）

两点需要注意：

1. **语料太小时指标会失真。** 16 个 chunk 取 top-4 等于覆盖 1/4 语料，
   所有策略都接近满分。用 `--top-k 1` 才能看出区分度——此时 bm25 单通道
   掉到 0.875，而 hybrid 保持 0.938。换成自己的真实语料（几百上千 chunk）
   区分度会自然显现。
2. `hit@k` 的上限是 15/16=0.938，因为第 16 题是故意放的库外问题
   （`expected_sources` 为空，按定义不计命中），它考察的是 `abstain_rate`。

### 5.10 单元测试

RAG 里真正容易出错的不是模型调用，而是去重、按来源删除、metadata 传递、
知识库枚举、跨通道去重、拒答判定这些管道细节。它们都能用假 embedding 完全覆盖，
因此不需要启动 LMStudio：

```bash
python -m unittest discover -s tests -v   # 42 项，约 1.6 秒
```

上表里带 ★ 的 bug 都有对应的回归用例，其中拒答判定和引用标记直接用真实跑出来的
答案文本做断言（`TestAbstainDetection`），避免同类误判再次发生。

## 各场景工作流详解

### 05 - 工具调用 (Tool Calling / Function Calling)

让大模型自主决定何时调用外部工具，是构建 Agent 的基础。

```
用户提问
    │
    ▼
LLM 分析问题 ──► 需要工具？──否──► 直接回答
    │
   是│
    ▼
选择工具 + 生成参数 (如 calculate({"expression": "123*456"}))
    │
    ▼
执行工具（本地 Python 函数）
    │
    ▼
将工具结果交给 LLM ──► 生成最终回答
```

注册了 4 个工具：`calculate`（计算器）、`get_current_time`（时间）、`roll_dice`（骰子）、`get_weather`（天气）。
模型通过工具的 docstring 理解何时该用哪个工具。

### 06 - 多智能体协作 (Multi-Agent Collaboration)

多个专职智能体协同完成复杂任务，用 LangGraph 编排协作与循环改进。

```
START
  │
  ▼
researcher(检索知识库) ──► writer(撰写初稿) ──► reviewer(审校打分)
                                                  │
                                            分数 >= 7 ?
                                           ┌──────┴──────┐
                                         是│            否│
                                          ▼              ▼
                                         END       writer(根据意见重写)
                                                        │
                                                        ▼
                                                    reviewer(再次审校)
```

- **研究 Agent**：从知识库检索信息，整理要点
- **写作 Agent**：基于研究结果撰写内容，可根据审校意见修改
- **审校 Agent**：审查内容，给出 0-10 分和修改意见
- 不达标则退回重写，最多修改 2 次

### 07 - Map-Reduce 长文档摘要

解决超长文档超出上下文窗口的问题，分而治之。

```
长文档
  │
  ▼
切分为 N 个文本块
  │
  ▼
┌────────┬────────┬────────┐
│ Map 1  │ Map 2  │ Map N  │   ← 每块独立生成摘要（可并行）
│ 摘要1  │ 摘要2  │ 摘要N  │
└────────┴────────┴────────┘
            │
            ▼
      合并所有摘要
            │
            ▼
      Reduce: 生成最终摘要
```

对比两种策略：
- **Stuff**：把全文塞进一个 prompt（适合短文档，1 次 LLM 调用）
- **Map-Reduce**：分块处理再合并（适合超长文档，N+1 次 LLM 调用）

> 注：LangChain 1.0+ 移除了 `load_summarize_chain`，本例用 LCEL 手动构建链。

### 08 - 条件路由工作流 (Router Workflow)

根据问题类型自动路由到不同处理链，是智能客服/问答系统的常见模式。

```
                     用户问题
                         │
                         ▼
                classify(LLM 分类)
                         │
          ┌──────────┬───┴────┬──────────┐
          ▼          ▼        ▼          ▼
       技术问题    创意写作  数学计算    闲聊
          │          │        │          │
          ▼          ▼        ▼          ▼
       RAG检索    创意写作  计算器工具   轻松回复
          │          │        │          │
          └──────────┴───┬────┴──────────┘
                         ▼
                      最终回答
```

四条分支各有专门处理逻辑：
- **技术问题** → RAG 链（从知识库检索回答）
- **创意写作** → 创意链（发挥想象力）
- **数学计算** → 工具链（调用计算器）
- **闲聊** → 对话链（轻松回复）

用 LangGraph 条件边实现动态分发，比 if-else 硬编码更清晰、更易扩展。
