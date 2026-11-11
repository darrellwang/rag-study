# RAG 学习环境部署全过程记录

> 记录时间：2026-06-24
> 环境：WSL (Linux) + Windows LMStudio 本地大模型
> 目标：部署 LangChain + LangGraph + RAG 学习环境，连接 Windows 上 LMStudio 提供的本地大模型

---

## 一、环境调研

### 1.1 检查 WSL 中 Python 环境

```bash
python3 --version   # Python 3.12.3
which python3       # /usr/bin/python3
pip3 --version      # pip 24.0
```

Python 3.12.3 可用，满足要求。

### 1.2 确定 Windows 宿主机 IP（WSL 访问 LMStudio）

WSL 访问 Windows 上的服务需要知道宿主机 IP，常见有几种方式：

```bash
# 方式1: 从 /etc/resolv.conf 获取
cat /etc/resolv.conf | grep nameserver   # 10.255.255.254

# 方式2: 从默认路由获取
ip route show | grep default             # 192.168.31.1

# 方式3: host.docker.internal（需要镜像网络模式）
getent hosts host.docker.internal        # 不可用
```

### 1.3 测试 LMStudio 连接

LMStudio 默认提供 OpenAI 兼容接口，端口 `1234`。测试发现 WSL 通过 `localhost:1234` 即可直连（说明当前 WSL 使用镜像网络模式）：

```bash
curl -s -m 3 http://localhost:1234/v1/models
```

返回的已加载模型清单（共 17 个）：

| 类型 | 模型 ID |
|------|---------|
| LLM | `qwen/qwen3.6-35b-a3b` |
| LLM | `unsloth/llama-3.3-70b-instruct` |
| LLM | `zai-org/glm-4.7-flash` |
| LLM | `qwen/qwen3-coder-30b` |
| LLM | `google/gemma-4-26b-a4b` |
| LLM | `qwen/qwen3.5-9b` |
| LLM | `qwen/qwen3-1.7b` |
| LLM | `meta/llama-3.3-70b` |
| LLM | `google/gemma-3-4b` |
| ... | (更多) |
| Embedding | `text-embedding-embeddinggemma-300m` |
| Embedding | `text-embedding-qwen3-embedding-0.6b` |
| Embedding | `text-embedding-nomic-embed-text-v1.5` |

选择默认配置：
- **对话模型**：`qwen/qwen3.6-35b-a3b`
- **Embedding 模型**：`text-embedding-embeddinggemma-300m`（768 维）

### 1.4 检查 WSL 网络模式

```bash
cat /etc/wsl.conf
# [boot]
# systemd=true
# [user]
# default=darrell
```

确认 WSL 为镜像网络模式（mirrored networking），因此 `localhost` 可直通 Windows。

---

## 二、安装依赖

### 2.1 创建虚拟环境

```bash
cd ~/workspace/rag-study
python3 -m venv .venv
```

### 2.2 安装 Python 包

```bash
.venv/bin/pip install --upgrade pip
.venv/bin/pip install \
  langchain langgraph langchain-community langchain-openai langchain-core \
  chromadb pypdf python-dotenv tiktoken
```

安装的核心包及版本：

| 包 | 版本 |
|----|------|
| langchain | 1.3.11 |
| langgraph | 1.2.6 |
| langchain-core | 1.4.8 |
| langchain-community | 0.4.2 |
| langchain-openai | 1.3.3 |
| langchain-text-splitters | 1.1.2 |
| chromadb | 1.5.9 |
| pypdf | 6.14.2 |
| python-dotenv | 1.2.2 |
| tiktoken | 0.13.0 |
| langchain-chroma | 1.1.0（后续补充安装） |

> `langchain-chroma` 是后来补充安装的，因为 `langchain_community.vectorstores.Chroma` 在 LangChain 1.0 后已弃用，会打印警告。改用独立包 `langchain-chroma` 后警告消除。

---

## 三、项目结构设计

```
rag-study/
├── .env                  # 模型与服务配置（实际使用）
├── .env.example          # 配置模板（可提交到 git）
├── .gitignore            # 忽略 .venv / .env / chroma_db / data 等
├── requirements.txt      # 依赖清单（含版本号）
├── README.md             # 使用说明
├── DEPLOYMENT_NOTES.md   # 本文件：部署全过程记录
├── data/                 # 知识库文档目录
│   ├── .gitkeep
│   └── example.txt       # 首次运行自动生成的示例文档
├── chroma_db/            # Chroma 向量库持久化数据（自动生成）
└── src/
    ├── config.py         # 集中配置：连接 LMStudio
    ├── rag_utils.py      # RAG 公共工具
    ├── 01_test_connection.py   # 连接测试
    ├── 02_basic_chat.py        # 基础多轮对话
    ├── 03_basic_rag.py         # 基础 RAG
    ├── 04_rag_langgraph.py     # LangGraph 纠错 RAG
    ├── 05_tool_use.py          # 工具调用 (Function Calling)
    ├── 06_multi_agent.py       # LangGraph 多智能体协作
    ├── 07_map_reduce_summarize.py  # Map-Reduce 长文档摘要
    └── 08_workflow_router.py   # LangGraph 条件路由工作流
```

---

## 四、核心配置文件

### 4.1 `.env`

```env
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio
LLM_MODEL=qwen/qwen3.6-35b-a3b
EMBEDDING_MODEL=text-embedding-embeddinggemma-300m
CHROMA_PERSIST_DIR=./chroma_db
DOCS_DIR=./data
```

> LMStudio 不需要真实 API Key，随便填一个占位即可。

### 4.2 `src/config.py` 关键设计

这是连接 LMStudio 的核心模块，解决了三个关键问题：

#### 问题 1：路径定位（跨目录运行）

```python
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

def _resolve(path: str) -> str:
    """相对路径基于项目根目录解析，绝对路径原样返回。"""
    p = Path(path)
    return str(p if p.is_absolute() else (BASE_DIR / p))
```

确保无论从哪个目录运行脚本，都能正确找到 `.env` 文件和 `data/`、`chroma_db/` 目录。

#### 问题 2：代理冲突（核心踩坑）

WSL 中配置了科学上网代理：
```bash
ALL_PROXY=socks5://127.0.0.1:7890
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
```

httpx 会尝试通过 SOCKS 代理访问 `localhost:1234`，导致报错：
```
Using SOCKS proxy, but the 'socksio' package is not installed.
```

虽然 `no_proxy` 已包含 `localhost`，但 openai SDK 仍走了代理。

**解决方案**：为本地连接创建 `trust_env=False` 的 httpx 客户端，绕过所有代理环境变量：

```python
import httpx

_HTTP_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
_local_http_client = httpx.Client(trust_env=False, timeout=_HTTP_TIMEOUT)
_local_async_http_client = httpx.AsyncClient(trust_env=False, timeout=_HTTP_TIMEOUT)

def get_llm(temperature=0.7, **kwargs):
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        http_client=_local_http_client,
        http_async_client=_local_async_http_client,
        **kwargs,
    )
```

#### 问题 3：Embedding 输入格式不兼容

`OpenAIEmbeddings` 默认 `check_embedding_ctx_length=True`，会把文本预先转成 token ID 数组发送。但 LMStudio 的 embedding 接口只接受字符串，报错：
```
Error code: 400 - {'error': "'input' field must be a string or an array of strings"}
```

**解决方案**：关闭该选项：
```python
def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        http_client=_local_http_client,
        http_async_client=_local_async_http_client,
        check_embedding_ctx_length=False,  # 关键
    )
```

#### 问题 4：超时设置

本地大模型推理较慢（尤其 70B 模型），HTTP 超时设为 300 秒：
```python
_HTTP_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
```

---

## 五、示例脚本设计

### 5.1 `01_test_connection.py` — 连接测试

- 测试对话模型：发送 "你好" 并检查回复
- 测试 embedding 模型：嵌入一段文本并检查向量维度
- 全部通过则输出 "环境就绪"

实测结果：
```
[1/2] 测试对话模型...
  回复: 我是通义千问（Qwen），由阿里巴巴集团...
  状态: OK
[2/2] 测试 Embedding 模型...
  向量维度: 768
  前 5 个值: [0.0723, 0.0616, 0.0156, -0.0134, -0.0074]
  状态: OK
所有测试通过，环境就绪！
```

### 5.2 `02_basic_chat.py` — 基础多轮对话

- 使用 `ChatPromptTemplate` + `MessagesPlaceholder` 管理对话历史
- 手动维护 `history` 列表（HumanMessage / AIMessage）
- 交互式输入，输入 `quit` 退出

实测验证了多轮记忆：
```
你: 我叫小明
AI: 你好小明！...
你: 我叫什么名字？
AI: 你叫小明。很高兴认识你！  # 正确记住上下文
```

### 5.3 `03_basic_rag.py` — 基础 RAG

流程：加载文档 → 切分 → 向量化存入 Chroma → 检索 → 生成

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

首次运行自动生成示例文档 `data/example.txt`（关于 LangChain/LangGraph/RAG 的介绍）。

实测结果：
```
构建向量数据库...
加载了 1 个文档
切分为 2 个文本块
问: LangChain 是什么?
[检索到 2 段相关文档]
答: LangChain 是一个用于开发由语言模型驱动的应用程序的框架...
```

### 5.4 `04_rag_langgraph.py` — LangGraph 纠错 RAG

实现了 **Corrective RAG (CRAG)** 工作流：

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

**图状态定义**：
```python
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retry_count: int
```

**四个节点**：
- `retrieve`：从向量库检索文档
- `grade`：用 LLM 逐篇评估文档相关性，过滤不相关的
- `generate`：基于相关文档生成回答
- `rewrite`：重写问题以便更好检索

**条件路由**：
```python
def decide_after_grade(state) -> str:
    if state["documents"]:
        return "generate"       # 有相关文档 → 生成
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "generate"       # 重试用完 → 兜底生成
    return "rewrite"            # 否则 → 重写问题
```

实测验证了完整纠错流程：
- 相关问题 "什么是 RAG？"：检索→评估(1/2相关)→生成正确答案
- 无关问题 "法国的首都是哪里？"：检索→评估(0/2)→重写→重新检索→重试2次用尽→兜底生成"巴黎"

### 5.5 `05_tool_use.py` — 工具调用 (Tool Calling)

场景：让大模型自主决定何时调用外部工具，这是构建 Agent 的基础——模型不再只是"说话"，而是能"做事"。

工作流：
```
用户提问 → LLM 分析是否需要工具
   否 → 直接回答
   是 → 选择工具 + 生成参数 → 执行工具(本地函数) → 将结果交给 LLM → 生成最终回答
```

**注册的 4 个工具**（用 `@tool` 装饰器，docstring 会被发送给模型作为使用依据）：

| 工具 | 功能 | 示例触发 |
|------|------|----------|
| `calculate` | 计算数学表达式（安全白名单校验） | "帮我算 123*456" |
| `get_current_time` | 获取当前日期时间 | "现在几点" |
| `roll_dice` | 掷骰子，可指定面数 | "掷个20面骰子" |
| `get_weather` | 模拟天气查询 | "北京天气" |

**核心代码**：
```python
tools = [calculate, get_current_time, roll_dice, get_weather]
llm_with_tools = llm.bind_tools(tools)   # 绑定工具

ai_msg = llm_with_tools.invoke([HumanMessage(content=user_input)])
if ai_msg.tool_calls:                     # 模型发起了工具调用
    for tc in ai_msg.tool_calls:
        tool_fn = {t.name: t for t in tools}[tc["name"]]
        result = tool_fn.invoke(tc["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    final = llm.invoke(messages)          # 基于工具结果生成最终回答
```

实测验证：模型能正确识别意图、选择合适工具、生成合法参数，并把工具返回结果整合进自然语言回答。

### 5.6 `06_multi_agent.py` — LangGraph 多智能体协作

场景：多个专职智能体协同完成复杂任务，体现 LangGraph 用图结构编排多角色协作 + 循环改进的核心价值。

本例实现一个"内容创作团队"，三个 Agent 分工：
- **研究 Agent**：基于本地知识库检索信息并整理要点
- **写作 Agent**：根据研究结果撰写初稿，或根据审校意见修改
- **审校 Agent**：审查草稿，给出意见和 0-10 分

工作流：
```
START → researcher(检索) → writer(初稿) → reviewer(审校打分)
                                                │
                                          分数 >= 7 ?
                                         ┌────────┴────────┐
                                        是│               否│
                                         ▼                  ▼
                                        END          writer(根据意见重写, 最多2次)
                                                            │
                                                            ▼
                                                       reviewer(再次审校)
```

**图状态**：
```python
class TeamState(TypedDict):
    task: str              # 用户给团队的任务
    research: str          # 研究员收集的信息
    draft: str             # 写手撰写的草稿
    review: str            # 审校员的意见
    score: int             # 审校打分 (0-10)
    revision_count: int    # 已修改次数
```

**条件路由**：分数 ≥7 或修改次数用尽(默认2次)则结束，否则退回 writer 重写。

> 踩坑提示：`reviewer` 节点之后**只能走条件边**，不能再加无条件边到 `writer`，否则条件边返回 `"end"` 时仍会继续走到 `writer`，造成死循环。

实测验证：对 "写一篇关于RAG的短文" 这类任务，团队能自动完成检索→撰写→审校→(若不达标)修改的完整闭环。

### 5.7 `07_map_reduce_summarize.py` — Map-Reduce 长文档摘要

场景：对超长文档生成摘要。单次把整篇文档丢给模型会超出上下文窗口，Map-Reduce 策略解决了这个问题。

工作流：
```
长文档 → 切分为 N 个文本块 → Map 阶段(每块独立摘要, 可并行) → Reduce 阶段(合并生成最终摘要)
```

本例对一篇约 1200 字的"AI 发展史"长文，对比两种摘要方式：

| 方式 | 做法 | 适用场景 | LLM 调用次数 |
|------|------|----------|--------------|
| **Stuff** | 把整篇塞进一个 prompt | 短文档 | 1 次 |
| **Map-Reduce** | 分块摘要再合并 | 超长文档 | N(map) + 1(reduce) |

**LCEL 手动构建链**（LangChain 1.0+ 已移除 `load_summarize_chain`，改用 LCEL 更透明）：
```python
# Map 阶段
map_chain = map_prompt | llm | StrOutputParser()
chunk_summaries = [map_chain.invoke({"text": c.page_content}) for c in chunks]

# Reduce 阶段
reduce_chain = reduce_prompt | llm | StrOutputParser()
final_summary = reduce_chain.invoke({"summaries": combined})
```

切分参数：`RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)`

实测验证：Stuff 与 Map-Reduce 均能生成合理摘要。对于真正的超长文档（如整本书），Stuff 会超出上下文窗口，而 Map-Reduce 能稳定工作——代价是更多 LLM 调用。若需提速，Map 阶段可用 `asyncio` 并行调用。

### 5.8 `08_workflow_router.py` — LangGraph 条件路由工作流

场景：根据用户问题类型自动路由到不同处理链，是构建智能客服、智能问答系统的常见模式。

本例实现"智能路由器"，将问题分为 4 类，分别由专门链处理：

| 类别 | 英文键 | 处理链 |
|------|--------|--------|
| 技术问题 | `tech` | RAG 链（从知识库检索回答） |
| 创意写作 | `creative` | 创意链（发挥想象力） |
| 数学计算 | `math` | 工具链（调用计算器工具） |
| 闲聊 | `chat` | 对话链（轻松随意） |

工作流：
```
用户问题 → classify(LLM 分类) ──┬── tech(技术)      ──┐
                                 ├── creative(创意)  ──┤
                                 ├── math(数学)      ──┼──→ END
                                 └── chat(闲聊)      ──┘
```

**图状态**：
```python
class RouterState(TypedDict):
    question: str       # 用户问题
    category: str       # 分类结果
    answer: str         # 最终回答
```

**条件边实现动态分发**：
```python
graph.add_conditional_edges(
    "classify",
    route_question,     # 返回 state["category"]
    {"tech": "tech", "creative": "creative", "math": "math", "chat": "chat"},
)
for node in ("tech", "creative", "math", "chat"):
    graph.add_edge(node, END)   # 所有分支汇聚到 END
```

分类节点用 LLM 输出类别英文名，并做容错提取（默认归为 `chat`）。`math` 分支复用了工具调用机制，让模型提取算式并调用 `calculate` 工具。

实测验证：对 "什么是RAG?"(tech)、"写一首关于AI的诗"(creative)、"算一下 999*888"(math)、"你好呀"(chat) 四类问题，路由器均能正确分类并调用对应链。

---

## 六、`src/rag_utils.py` — RAG 公共工具

被 `03` 和 `04` 脚本共用，提供：

1. **`build_or_load_vectorstore()`**：智能构建/加载向量库
   - 已有持久化数据 → 直接加载（避免重复嵌入）
   - 无数据 → 从 `data/` 加载 `.txt` 文档 → 切分 → 写入 Chroma
   - 无文档 → 自动生成示例文档

2. **`format_docs()`**：把多个文档拼接为上下文字符串

3. **内置示例知识库**：关于 LangChain/LangGraph/RAG/向量数据库的中文介绍文本

文档切分参数：
```python
RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
```

---

## 七、验证测试

所有 8 个脚本均实测通过：

| 脚本 | 测试内容 | 结果 |
|------|----------|------|
| `01_test_connection.py` | 对话 + Embedding 连接 | ✓ 通过 |
| `02_basic_chat.py` | 多轮对话记忆 | ✓ 通过 |
| `03_basic_rag.py` | 向量库构建 + 检索生成 | ✓ 通过 |
| `04_rag_langgraph.py` | 纠错 RAG 完整流程 | ✓ 通过 |
| `05_tool_use.py` | 工具调用 (计算器/时间/骰子/天气) | ✓ 通过 |
| `06_multi_agent.py` | 多智能体协作 + 循环修改 | ✓ 通过 |
| `07_map_reduce_summarize.py` | Stuff vs Map-Reduce 摘要 | ✓ 通过 |
| `08_workflow_router.py` | 条件路由 4 分支工作流 | ✓ 通过 |

无 lint 错误，无弃用警告。

---

## 八、踩坑总结

### 坑 1：SOCKS 代理拦截 localhost 请求

- **现象**：`Using SOCKS proxy, but the 'socksio' package is not installed`
- **原因**：WSL 配置了 `ALL_PROXY=socks5://127.0.0.1:7890`，httpx 尝试经代理访问 localhost
- **解决**：httpx 客户端设 `trust_env=False`

### 坑 2：Embedding 请求格式不兼容

- **现象**：`400 - 'input' field must be a string or an array of strings`
- **原因**：`OpenAIEmbeddings` 默认把文本转成 token ID 数组，LMStudio 只接受字符串
- **解决**：设 `check_embedding_ctx_length=False`

### 坑 3：Chroma 弃用警告

- **现象**：`langchain_community.vectorstores.Chroma was deprecated`
- **原因**：LangChain 1.0 后 Chroma 迁移到独立包
- **解决**：`pip install langchain-chroma`，改用 `from langchain_chroma import Chroma`

### 坑 4：路径定位问题

- **现象**：从非项目根目录运行脚本时找不到 `.env` 或 `data/`
- **解决**：`config.py` 中基于 `__file__` 定位项目根目录，所有相对路径自动解析

---

## 九、如何复现

```bash
# 1. 确保 Windows 上 LMStudio 已启动 Server (端口 1234)
#    并加载了对话模型和 embedding 模型

# 2. 在 WSL 中
cd ~/workspace/rag-study
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. 测试连接
python src/01_test_connection.py

# 4. 体验各示例
python src/02_basic_chat.py
python src/03_basic_rag.py
python src/04_rag_langgraph.py
python src/05_tool_use.py
python src/06_multi_agent.py
python src/07_map_reduce_summarize.py
python src/08_workflow_router.py
```

---

## 十、扩展指南

### 添加自己的知识库

```bash
rm -rf chroma_db                    # 删除旧向量库
cp /path/to/your.pdf data/          # 放入文档
python src/03_basic_rag.py          # 重新构建并问答
```

### 切换模型

编辑 `.env`，修改 `LLM_MODEL` 或 `EMBEDDING_MODEL` 为 LMStudio 中已加载的其他模型。
查看可用模型：
```bash
curl -s http://localhost:1234/v1/models | python3 -m json.tool
```

### 如果 WSL 不是镜像网络模式

若 `localhost:1234` 无法连通，需使用 Windows 宿主机 IP：
```bash
# 获取宿主机 IP
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
# 然后在 .env 中修改
# LMSTUDIO_BASE_URL=http://<宿主机IP>:1234/v1
```
同时需在 LMStudio 中开启 "Serve on local network" 选项。
