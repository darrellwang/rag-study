"""
集中配置模块：连接 LMStudio 提供的本地大模型。

LMStudio 暴露的是 OpenAI 兼容接口，因此可以直接使用
langchain-openai 的 ChatOpenAI / OpenAIEmbeddings，
只需把 base_url 指向 LMStudio 即可。
"""
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

# 项目根目录 = src/ 的上一级
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载项目根目录下的 .env（无论从哪个目录运行都能找到）
_ = load_dotenv(BASE_DIR / ".env")


def _resolve(path: str) -> str:
    """相对路径基于项目根目录解析，绝对路径原样返回。"""
    p = Path(path)
    return str(p if p.is_absolute() else (BASE_DIR / p))


# ---- 基础配置 ----
BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = SecretStr(os.getenv("LMSTUDIO_API_KEY", "lm-studio"))
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.6-35b-a3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-embeddinggemma-300m")
CHROMA_PERSIST_DIR = _resolve(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
DOCS_DIR = _resolve(os.getenv("DOCS_DIR", "./data"))
# 阶段4：可选的服务端 API Key 鉴权（为空则不校验）
SERVICE_API_KEY = os.getenv("KB_API_KEY", "")

# ---- 切分与检索参数 ----
# chunk_size 以字符计。中文语义密度高于英文，300~800 是常见区间：
# 太小丢上下文，太大则一个 chunk 里混入多个主题、拉低向量区分度。
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# 最终交给大模型的片段数
TOP_K = int(os.getenv("TOP_K", "4"))
# 召回阶段的候选数（> TOP_K，留给重排/融合筛选的空间）
FETCH_K = int(os.getenv("FETCH_K", "20"))

# 混合检索中向量通道的权重（BM25 通道为 1 - 该值）
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.6"))
# RRF 融合的平滑常数，越大则名次差异被压缩得越平
RRF_K = int(os.getenv("RRF_K", "60"))

# 默认检索策略：vector | mmr | bm25 | hybrid
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
# 是否默认启用 LLM 重排（本地小模型下会明显变慢，默认关闭）
USE_RERANK = os.getenv("USE_RERANK", "false").lower() in ("1", "true", "yes")

# ---- 外部连接器凭据（均为空时进入 MOCK 模式）----
# Confluence
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL", "")
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL", "")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", "")
CONFLUENCE_SPACES = [s.strip() for s in os.getenv("CONFLUENCE_SPACES", "").split(",") if s.strip()]
# Jira
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
JIRA_JQL = os.getenv("JIRA_JQL", "ORDER BY updated DESC")
# GitHub
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPOS = [r.strip() for r in os.getenv("GITHUB_REPOS", "").split(",") if r.strip()]
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

# LMStudio 运行在本机 localhost，必须绕过系统代理。
# WSL 中常配置了 ALL_PROXY=socks5://... 用来科学上网，若不关闭，
# httpx 会尝试经 SOCKS 代理访问 localhost 而报错。
# trust_env=False 让 httpx 忽略 *_PROXY 环境变量，直连本地服务。
# 同时设置较长超时（本地大模型推理可能较慢，尤其 70B 模型）。
_HTTP_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
_local_http_client = httpx.Client(trust_env=False, timeout=_HTTP_TIMEOUT)
_local_async_http_client = httpx.AsyncClient(trust_env=False, timeout=_HTTP_TIMEOUT)


def get_llm(temperature: float = 0.7, **kwargs) -> ChatOpenAI:
    """返回连接 LMStudio 的对话大模型。"""
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        http_client=_local_http_client,
        http_async_client=_local_async_http_client,
        **kwargs,
    )


# 打分模型是否支持关闭思维链。None = 还没探测过
_SCORING_NO_THINK: bool | None = None


def get_scoring_llm(temperature: float = 0.0) -> ChatOpenAI:
    """返回用于「打分/分类」的大模型，尽量关闭思维链。

    为什么需要单独一个：
    重排和文档相关性评估本质是分类任务，期望输出只有一个数字或 yes/no。
    但现在主流的本地模型（Qwen3 系列等）默认开启思维链，为了输出一个 "10"
    会先生成 300~2500 个隐藏的 reasoning token。实测同一次打分：
        开思考 48.2s（2543 output tokens，其中 2539 是 reasoning）
        关思考  4.9s（2 output tokens）
    差了一个数量级，而打分结果完全一致。重排要跑 fetch_k 次，
    这个差异会直接决定「重排能不能用」。

    OpenAI 兼容接口里通过 reasoning_effort="none" 关闭（LMStudio 支持）。
    注意 LMStudio 不认 extra_body 里的 chat_template_kwargs.enable_thinking。
    并非所有服务端都支持该参数，因此这里做一次性探测并缓存结果，
    不支持时安静退回普通模型（只是慢，不会出错）。
    """
    global _SCORING_NO_THINK

    if _SCORING_NO_THINK is None:
        try:
            probe = ChatOpenAI(
                model=LLM_MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0,
                http_client=_local_http_client, http_async_client=_local_async_http_client,
                reasoning_effort="none", max_completion_tokens=16,
            )
            probe.invoke("回答一个字：好")
            _SCORING_NO_THINK = True
        except Exception as e:
            print(f"  [info] 当前模型/服务不支持 reasoning_effort=none（{type(e).__name__}），"
                  "打分类调用将保留思维链，重排会明显偏慢")
            _SCORING_NO_THINK = False

    kwargs = {"reasoning_effort": "none"} if _SCORING_NO_THINK else {}
    return get_llm(temperature=temperature, **kwargs)


def get_embeddings() -> OpenAIEmbeddings:
    """返回连接 LMStudio 的 embedding 模型。"""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        http_client=_local_http_client,
        http_async_client=_local_async_http_client,
        # 关闭 token 长度检查：默认会把文本转成 token ID 数组发送，
        # 但 LMStudio 的 embedding 接口只接受字符串，会报 400 错误。
        check_embedding_ctx_length=False,
    )
