"""
端到端 RAG 管道：一个配置对象决定整条链路的行为。

相比 03/04 脚本里手写的链，这里多做了三件生产环境必需的事：

1. 带编号的上下文 + 强制引用。
   上下文以 [1] [2] 的形式编号后交给模型，并要求答案里标注引用来源。
   好处不只是好看：一旦要求逐句标引用，模型编造的成本就变高了，
   而且用户能直接定位原文核对，这是 RAG 可信度的地基。

2. 拒答（abstain）。
   检索为空、或重排后最高分低于门槛时，直接返回「知识库中没有相关信息」，
   不进入生成。宁可不答，也不要一本正经地胡说。

3. 耗时分解。
   retrieve / rerank / generate 各自耗时都记录下来。
   没有这个数字，性能优化就只能靠猜——绝大多数情况下慢的是重排和改写，
   不是向量检索。

流式：stream() 走真正的 token 级流式（llm.stream），
而不是先算完整答案再切片假装流式。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import FETCH_K, RETRIEVAL_MODE, TOP_K, USE_RERANK, get_llm
from .retriever import HybridRetriever
from .store import KnowledgeBase, get_kb

ABSTAIN_MESSAGE = "知识库中没有找到与该问题相关的信息，无法回答。"

# 模型按提示词拒答时会说「资料不足，无法回答」。识别出来才能把它计入拒答率，
# 也才能避免给一个「无法回答」配上一串无关来源 —— 那比不给来源更误导。
_ABSTAIN_MARKERS = ("资料不足", "无法回答", "没有相关信息", "未找到相关")

# 只靠关键词判断会误伤：问「为什么要有拒答机制」时，正确答案本身就包含
# 「无法回答」「没有相关信息」这些词（实测被误判过）。所以再加两个条件：
#   1. 真拒答很短（实测 9 字符 vs 误判样本 164 字符）
#   2. 真拒答不会标引用，而基于资料的回答会带 [1][2]
_ABSTAIN_MAX_CHARS = 60

_CITATION_REF = re.compile(r"\[(\d+)\]")


def mark_used_citations(answer: str, citations: List[dict]) -> List[dict]:
    """标记哪些来源在答案里被真正引用过。

    检索了 4 条但只用了 1 条是常态。把全部 4 条都当作「依据」展示给用户，
    会让引用失去意义 —— 用户点开发现内容跟答案无关，就再也不信引用了。
    """
    referenced = {int(n) for n in _CITATION_REF.findall(answer or "")}
    for c in citations:
        c["used"] = c["index"] in referenced
    return citations


def looks_abstained(answer: str) -> bool:
    """判断模型是否在拒答。三个条件必须同时满足，避免误伤正常答案。"""
    text = (answer or "").strip()
    if not text or len(text) > _ABSTAIN_MAX_CHARS:
        return False
    if _CITATION_REF.search(text):  # 标了引用就说明用上了资料
        return False
    return any(marker in text for marker in _ABSTAIN_MARKERS)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "你是严谨的知识库问答助手。请严格依据下面带编号的资料回答问题。\n"
    "规则：\n"
    "1. 只使用资料中的信息，不得引入资料之外的内容。\n"
    "2. 在使用了某条资料的句子末尾标注来源编号，如 [1]、[2][3]。\n"
    "3. 若资料不足以回答，直接说明「资料不足，无法回答」，不要猜测。\n"
    "4. 用简洁的中文回答。\n"
    "{history_block}"
    "\n资料:\n{context}\n\n问题: {question}\n\n回答:"
)


@dataclass
class RagConfig:
    """一处改配置，整条链路跟着变，便于做 A/B。"""

    kb_id: str = "default"
    mode: str = RETRIEVAL_MODE          # vector | mmr | bm25 | hybrid
    top_k: int = TOP_K
    fetch_k: int = FETCH_K
    fusion: str = "rrf"                 # rrf | weighted
    use_rerank: bool = USE_RERANK
    use_multi_query: bool = False
    use_hyde: bool = False
    temperature: float = 0.0
    metadata_filter: Optional[dict] = None
    allow_abstain: bool = True

    def label(self) -> str:
        """给评估报告用的简短配置名。必须能区分所有影响结果的开关，
        否则 A/B 表格里会出现两行同名配置，根本没法比。"""
        name = self.mode
        if self.mode == "hybrid" and self.fusion != "rrf":
            name += f"-{self.fusion}"
        flags = []
        if self.use_rerank:
            flags.append("rerank")
        if self.use_multi_query:
            flags.append("mq")
        if self.use_hyde:
            flags.append("hyde")
        suffix = ("+" + "+".join(flags)) if flags else ""
        return f"{name}(k={self.top_k}){suffix}"


@dataclass
class RagAnswer:
    question: str
    answer: str
    citations: List[dict] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    abstained: bool = False
    config_label: str = ""
    queries: List[str] = field(default_factory=list)

    @property
    def used_citations(self) -> List[dict]:
        """答案里真正引用到的来源。"""
        return [c for c in self.citations if c.get("used")]

    def pretty(self) -> str:
        lines = [f"答: {self.answer}", ""]
        if self.citations:
            lines.append("引用（✓ = 答案中实际引用）:")
            for c in self.citations:
                mark = "✓" if c.get("used") else " "
                loc = f" p.{c['page']}" if c.get("page") is not None else ""
                lines.append(f"  {mark} [{c['index']}] {c['source']}{loc}")
        lines.append(f"耗时(ms): {self.timings_ms}")
        return "\n".join(lines)


def build_context(docs: List[Document]) -> tuple:
    """把文档拼成带编号的上下文，同时产出引用清单。

    编号必须和引用清单严格一一对应，否则模型标的 [2] 会指向错误来源。
    """
    blocks, citations = [], []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        source = meta.get("file_name") or meta.get("source") or "unknown"
        heading = meta.get("heading_path")
        header = f"[{i}] 来源: {source}"
        if heading:
            header += f" | 章节: {heading}"
        if meta.get("page") is not None:
            header += f" | 第 {meta['page']} 页"
        blocks.append(f"{header}\n{d.page_content}")
        citations.append(
            {
                "index": i,
                "source": str(source),
                "kb_id": meta.get("kb_id", ""),
                "heading_path": heading,
                "page": meta.get("page"),
                "snippet": d.page_content[:200],
            }
        )
    return "\n\n---\n\n".join(blocks), citations


class RagPipeline:
    def __init__(
        self,
        config: Optional[RagConfig] = None,
        kb: Optional[KnowledgeBase] = None,
        llm=None,
        scoring_llm=None,
    ):
        self.config = config or RagConfig()
        self.llm = llm or get_llm(temperature=self.config.temperature)
        # 打分/重排用关闭思维链的实例，与生成用的模型分开
        self.scoring_llm = scoring_llm
        self.kb = kb or get_kb(self.config.kb_id)
        self.retriever = self._build_retriever()

    def _build_retriever(self) -> HybridRetriever:
        c = self.config
        return HybridRetriever(
            kb=self.kb,
            mode=c.mode,
            top_k=c.top_k,
            fetch_k=c.fetch_k,
            fusion=c.fusion,
            use_rerank=c.use_rerank,
            use_multi_query=c.use_multi_query,
            use_hyde=c.use_hyde,
            llm=self.llm,
            scoring_llm=self.scoring_llm,
            metadata_filter=c.metadata_filter,
        )

    def reconfigure(self, config: RagConfig) -> "RagPipeline":
        """换配置但复用同一个 KB / LLM 连接，评估时反复切换配置很省事。"""
        self.config = config
        self.kb = get_kb(config.kb_id)
        self.retriever = self._build_retriever()
        return self

    def invalidate(self) -> None:
        self.retriever.invalidate()

    def _prompt_inputs(self, question: str, docs: List[Document], history: str) -> dict:
        context, citations = build_context(docs)
        history_block = f"\n参考对话历史（仅用于理解指代关系，不作为事实来源）:\n{history}\n" if history else ""
        return {
            "inputs": {"context": context, "question": question, "history_block": history_block},
            "citations": citations,
        }

    def answer(self, question: str, history: str = "") -> RagAnswer:
        retrieval = self.retriever.search(question)
        timings = dict(retrieval.timings_ms)

        if not retrieval.documents and self.config.allow_abstain:
            return RagAnswer(
                question=question,
                answer=ABSTAIN_MESSAGE,
                timings_ms=timings,
                abstained=True,
                config_label=self.config.label(),
                queries=retrieval.queries,
            )

        prepared = self._prompt_inputs(question, retrieval.documents, history)
        chain = ANSWER_PROMPT | self.llm | StrOutputParser()

        t = time.perf_counter()
        answer_text = chain.invoke(prepared["inputs"]).strip()
        timings["generate"] = round((time.perf_counter() - t) * 1000, 1)
        timings["total"] = round(timings.get("total", 0) + timings["generate"], 1)

        abstained = looks_abstained(answer_text)
        citations = [] if abstained else mark_used_citations(answer_text, prepared["citations"])

        return RagAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            documents=retrieval.documents,
            timings_ms=timings,
            abstained=abstained,
            config_label=self.config.label(),
            queries=retrieval.queries,
        )

    def stream(self, question: str, history: str = "") -> Iterator[dict]:
        """真流式。事件类型：
        retrieved  检索完成，带引用清单（前端可先渲染来源）
        token      生成的增量文本
        done       结束，带耗时
        """
        retrieval = self.retriever.search(question)
        timings = dict(retrieval.timings_ms)

        if not retrieval.documents and self.config.allow_abstain:
            yield {"type": "retrieved", "citations": []}
            yield {"type": "token", "text": ABSTAIN_MESSAGE}
            yield {"type": "done", "timings_ms": timings, "abstained": True}
            return

        prepared = self._prompt_inputs(question, retrieval.documents, history)
        yield {"type": "retrieved", "citations": prepared["citations"]}

        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        t = time.perf_counter()
        collected = []
        for piece in chain.stream(prepared["inputs"]):
            if piece:
                collected.append(piece)
                yield {"type": "token", "text": piece}
        timings["generate"] = round((time.perf_counter() - t) * 1000, 1)

        # 哪些来源被真正引用，只有生成完才知道。meta 里已经把全部来源发过去了，
        # 这里补发实际引用编号，前端可以把没用上的置灰。
        answer_text = "".join(collected)
        used = [c["index"] for c in mark_used_citations(answer_text, prepared["citations"]) if c["used"]]
        yield {
            "type": "done",
            "timings_ms": timings,
            "abstained": looks_abstained(answer_text),
            "used_citations": used,
        }
