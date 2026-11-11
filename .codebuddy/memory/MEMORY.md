# 长期记忆

## 项目: rag-study
- 路径: `/home/darrell/workspace/rag-study/`
- 用途: LangChain + LangGraph + RAG 学习环境
- Python 虚拟环境: `.venv/`（Python 3.12.3）
- 连接 Windows 上 LMStudio 本地大模型（OpenAI 兼容接口 `localhost:1234/v1`）

## 用户环境
- OS: WSL (Linux)，镜像网络模式，可通过 localhost 直连 Windows 服务
- 代理: WSL 配置了 `ALL_PROXY=socks5://127.0.0.1:7890`（科学上网），访问 localhost 需绕过
- LMStudio 默认模型: 对话 `qwen/qwen3.6-35b-a3b`，Embedding `text-embedding-embeddinggemma-300m`

## 关键经验
- WSL 中 httpx 访问 localhost 时需设 `trust_env=False` 绕过系统代理
- LMStudio 的 embedding 接口只接受字符串，`OpenAIEmbeddings` 需关闭 `check_embedding_ctx_length`
- LangChain 1.0+ 的 Chroma 应使用独立包 `langchain-chroma`，而非 `langchain_community.vectorstores`
- 本地大模型推理慢，HTTP 超时应设 300 秒
