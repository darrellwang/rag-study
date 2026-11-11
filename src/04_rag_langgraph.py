"""
04 - 进阶 RAG with LangGraph：Corrective RAG（带自我纠错的检索增强生成）

运行:  python 04_rag_langgraph.py

工作流（有向图）:
    START
      │
      ▼
    retrieve(检索) ──► grade(评估文档相关性)
                            │
                    相关文档 >= 1 ?
                   ┌────────┴────────┐
                 是│                 否│
                   ▼                  ▼
               generate            rewrite(重写问题)
                   │                  │
                   ▼                  └──► retrieve (重新检索, 最多重试 2 次)
                  END

当检索到的文档都不相关时，LangGraph 让模型重写问题后重新检索，
而不是硬编答案，从而减少幻觉——这正是用图结构编排循环的价值。
"""
from typing import Any

from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI

from config import get_llm
from rag_utils import build_or_load_vectorstore, format_docs

MAX_RETRIES = 2  # 问题重写的最大重试次数，防止无限循环


# ===== 1. 定义图的状态（在节点间流转的数据） =====
class GraphState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    retry_count: int


# ===== 2. 节点函数 =====
def retrieve(state: GraphState, retriever: Any) -> dict[str, Any]:
    """检索与问题相关的文档。"""
    question = state["question"]
    docs = retriever.invoke(question)
    print(f"  [retrieve] 检索到 {len(docs)} 篇文档")
    return {"documents": docs, "question": question}


def grade_documents(state: GraphState, llm: ChatOpenAI) -> dict[str, Any]:
    """逐篇评估文档与问题的相关性，过滤掉不相关的。"""
    question = state["question"]
    docs = state["documents"]

    prompt = ChatPromptTemplate.from_template(
        "你是一个评估文档相关性的助手。请判断下面的文档是否与问题相关。\n"
        "只回答 'yes' 或 'no'。\n\n问题: {question}\n文档: {document}\n判断:"
    )
    chain = prompt | llm | StrOutputParser()

    filtered = []
    for d in docs:
        result = chain.invoke(
            {"question": question, "document": d.page_content}
        ).strip().lower()
        if result.startswith("yes"):
            filtered.append(d)

    print(f"  [grade] {len(filtered)}/{len(docs)} 篇文档相关")
    return {"documents": filtered, "question": question}


def generate(state: GraphState, llm: ChatOpenAI) -> dict[str, Any]:
    """基于检索到的文档生成最终回答。"""
    question = state["question"]
    docs = state["documents"]

    prompt = ChatPromptTemplate.from_template(
        "请根据以下上下文回答问题。若上下文不足以回答，请明确说明。\n\n"
        "上下文:\n{context}\n\n问题: {question}\n回答:"
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    return {"documents": docs, "question": question, "generation": answer}


def rewrite_query(state: GraphState, llm: ChatOpenAI) -> dict[str, Any]:
    """重写问题，使其更适合检索。"""
    question = state["question"]
    prompt = ChatPromptTemplate.from_template(
        "请用更好的措辞重写下面的问题，使其更适合用于文档检索。"
        "只输出重写后的问题，不要解释。\n原问题: {question}\n重写后:"
    )
    chain = prompt | llm | StrOutputParser()
    new_q = chain.invoke({"question": question}).strip()
    print(f"  [rewrite] 重写问题: {new_q}")
    return {
        "question": new_q,
        "documents": state["documents"],
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ===== 3. 路由函数（决定下一步走哪条边） =====
def decide_after_grade(state: GraphState) -> str:
    """有相关文档则生成；否则重写问题（重试次数用完则兜底生成）。"""
    if state["documents"]:
        return "generate"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        print("  [route] 重试次数用完，兜底生成")
        return "generate"
    return "rewrite"


# ===== 4. 构建并编译图 =====
def build_graph(retriever: Any, llm: ChatOpenAI):
    graph = StateGraph(GraphState)

    # 添加节点（lambda 捕获额外参数与 LangGraph 类型签名不兼容，忽略此处类型检查）
    graph.add_node("retrieve", lambda s: retrieve(s, retriever))  # pyright: ignore[reportArgumentType]
    graph.add_node("grade", lambda s: grade_documents(s, llm))  # pyright: ignore[reportArgumentType]
    graph.add_node("generate", lambda s: generate(s, llm))  # pyright: ignore[reportArgumentType]
    graph.add_node("rewrite", lambda s: rewrite_query(s, llm))  # pyright: ignore[reportArgumentType]

    # 添加边（固定走向）
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("generate", END)
    graph.add_edge("rewrite", "retrieve")  # 重写后回到检索

    # 条件边：评估后根据相关性分流
    graph.add_conditional_edges(
        "grade",
        decide_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )

    return graph.compile()


# ===== 5. 主循环 =====
def main():
    print("LangGraph Corrective RAG 启动中...")
    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm(temperature=0)
    app = build_graph(retriever, llm)

    print("\n" + "=" * 55)
    print("  LangGraph RAG 已就绪（输入 quit 退出）")
    print("=" * 55 + "\n")

    while True:
        question = input("问: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not question:
            continue

        result = app.invoke(
            {"question": question, "documents": [], "generation": "", "retry_count": 0}
        )
        print(f"\n答: {result['generation']}\n")


if __name__ == "__main__":
    main()
