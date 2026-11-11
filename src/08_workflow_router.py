"""
08 - LangGraph 条件路由工作流 (Router Workflow)

运行:  python 08_workflow_router.py

场景: 根据用户问题的类型，自动路由到不同的处理链。
      这是构建智能客服、智能问答系统的常见模式。

本例实现一个"智能路由器"，将问题分为 4 类，分别由专门的链处理:
    - 技术问题  → RAG 链（从知识库检索回答）
    - 创意写作  → 创意链（发挥想象力，不限知识库）
    - 数学计算  → 工具链（调用计算器工具）
    - 闲聊      → 对话链（轻松随意的回复）

工作流:
                         用户问题
                             │
                             ▼
                    classifier(分类)
                             │
              ┌──────────┬───┴────┬──────────┐
              ▼          ▼        ▼          ▼
           技术问题    创意写作  数学计算    闲聊
              │          │        │          │
              ▼          ▼        ▼          ▼
           rag_chain  creative  calc_tool   chat
              │          │        │          │
              └──────────┴───┬────┴──────────┘
                             ▼
                          合并结果
                             │
                             ▼
                            END

体现 LangGraph 价值: 用条件边实现动态分发，
比 if-else 硬编码更清晰、更易扩展新类别。
"""
from typing import Any

from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config import get_llm
from rag_utils import build_or_load_vectorstore, format_docs


# ===== 1. 图状态 =====
class RouterState(TypedDict):
    question: str       # 用户问题
    category: str       # 分类结果
    answer: str         # 最终回答


# ===== 2. 分类节点 =====
CATEGORIES = {"tech": "技术问题", "creative": "创意写作", "math": "数学计算", "chat": "闲聊"}


def classify_node(state: RouterState, llm: ChatOpenAI) -> dict[str, Any]:
    """用 LLM 对问题分类。"""
    question = state["question"]
    prompt = ChatPromptTemplate.from_template(
        "请将用户问题分类为以下类别之一，只输出类别英文名:\n"
        "- tech: 技术问题（编程、AI、计算机等）\n"
        "- creative: 创意写作（写故事、诗歌、文案等）\n"
        "- math: 数学计算（需要数值计算的）\n"
        "- chat: 闲聊（问候、日常对话）\n\n"
        "问题: {question}\n类别:"
    )
    chain = prompt | llm | StrOutputParser()
    category = chain.invoke({"question": question}).strip().lower()

    # 容错: 提取第一个匹配的关键词
    for key in CATEGORIES:
        if key in category:
            category = key
            break
    else:
        category = "chat"  # 默认归为闲聊

    print(f"  [分类] -> {category} ({CATEGORIES[category]})")
    return {"category": category}


# ===== 3. 各类别的处理节点 =====
def tech_node(state: RouterState, retriever: Any, llm: ChatOpenAI) -> dict[str, Any]:
    """技术问题: 用 RAG 从知识库检索回答。"""
    question = state["question"]
    docs = retriever.invoke(question)
    context = format_docs(docs) if docs else "（无相关资料）"
    print(f"  [技术链] 检索到 {len(docs)} 篇文档")

    prompt = ChatPromptTemplate.from_template(
        "你是技术专家。请根据以下资料回答技术问题。\n\n资料:\n{context}\n\n问题: {question}\n回答:"
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer}


def creative_node(state: RouterState, llm: ChatOpenAI) -> dict[str, Any]:
    """创意写作: 发挥想象力。"""
    question = state["question"]
    msg = llm.invoke([
        SystemMessage(content="你是一位富有想象力的创意作家。请用优美的文笔完成用户的写作请求。"),
        HumanMessage(content=question),
    ])
    return {"answer": msg.content}


def math_node(state: RouterState, llm: ChatOpenAI) -> dict[str, Any]:
    """数学计算: 调用计算器工具。"""
    question = state["question"]

    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式，支持加减乘除和括号。"""
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expression):
            return "错误: 表达式包含非法字符"
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"计算失败: {e}"

    tools = [calculate]
    llm_with_tools = llm.bind_tools(tools)

    # 让模型提取算式并调用计算器
    ai_msg = llm_with_tools.invoke([
        SystemMessage(content="你是数学助手。请提取用户问题中的数学表达式并调用计算工具。"),
        HumanMessage(content=question),
    ])

    if ai_msg.tool_calls:
        from langchain_core.messages import ToolMessage

        messages = [HumanMessage(content=question), ai_msg]
        for tc in ai_msg.tool_calls:
            print(f"  [数学链] 调用计算器: {tc['args']}")
            result = calculate.invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        final = llm.invoke(messages)
        return {"answer": final.content}

    return {"answer": ai_msg.content}


def chat_node(state: RouterState, llm: ChatOpenAI) -> dict[str, Any]:
    """闲聊: 轻松随意。"""
    question = state["question"]
    msg = llm.invoke([
        SystemMessage(content="你是一个友好的聊天伙伴。请轻松自然地回复。"),
        HumanMessage(content=question),
    ])
    return {"answer": msg.content}


# ===== 4. 路由函数 =====
def route_question(state: RouterState) -> str:
    """根据分类结果路由到对应节点。"""
    return state["category"]


# ===== 5. 构建图 =====
def build_router_graph(retriever: Any, llm: ChatOpenAI):
    graph = StateGraph(RouterState)

    graph.add_node("classify", lambda s: classify_node(s, llm))
    graph.add_node("tech", lambda s: tech_node(s, retriever, llm))
    graph.add_node("creative", lambda s: creative_node(s, llm))
    graph.add_node("math", lambda s: math_node(s, llm))
    graph.add_node("chat", lambda s: chat_node(s, llm))

    # 入口 → 分类
    graph.add_edge(START, "classify")

    # 分类后根据结果路由到 4 个分支之一
    graph.add_conditional_edges(
        "classify",
        route_question,
        {"tech": "tech", "creative": "creative", "math": "math", "chat": "chat"},
    )

    # 所有分支汇聚到 END
    for node in ("tech", "creative", "math", "chat"):
        graph.add_edge(node, END)

    return graph.compile()


# ===== 6. 主逻辑 =====
def main():
    print("智能路由工作流启动中...")
    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm(temperature=0.3)
    router = build_router_graph(retriever, llm)

    print("\n" + "=" * 60)
    print("  智能路由器已就绪（输入 quit 退出）")
    print("  支持分类: 技术问题 / 创意写作 / 数学计算 / 闲聊")
    print("  试试: '什么是RAG?' / '写一首关于AI的诗' / '算一下 999*888' / '你好呀'")
    print("=" * 60 + "\n")

    while True:
        question = input("问: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not question:
            continue

        result = router.invoke({"question": question, "category": "", "answer": ""})
        print(f"\n答: {result['answer']}\n")


if __name__ == "__main__":
    main()
