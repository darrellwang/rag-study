"""
06 - LangGraph 多智能体协作 (Multi-Agent Collaboration)

运行:  python 06_multi_agent.py

场景: 多个专职智能体协同完成复杂任务。每个 Agent 有自己的角色和工具，
      通过 LangGraph 编排它们之间的协作流程。

本例实现一个"内容创作团队":
    - 研究 Agent: 负责收集信息（基于本地知识库检索）
    - 写作 Agent: 负责撰写内容
    - 审校 Agent: 负责检查并打分，不合格则退回重写

工作流:
    START
      │
      ▼
    researcher(检索知识库) ──► writer(撰写初稿) ──► reviewer(审校打分)
                                                      │
                                                分数 >= 7 ?
                                               ┌──────┴──────┐
                                             是│            否│
                                              ▼              ▼
                                             END       writer(根据意见重写, 最多2次)
                                                            │
                                                            ▼
                                                        reviewer(再次审校)

体现 LangGraph 核心价值: 用图结构编排多角色协作 + 循环改进。
"""
from typing import Any

from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config import get_llm
from rag_utils import build_or_load_vectorstore

MAX_REVISIONS = 2  # 最大修改次数


# ===== 1. 图状态 =====
class TeamState(TypedDict):
    task: str              # 用户给团队的任务
    research: str          # 研究员收集的信息
    draft: str             # 写手撰写的草稿
    review: str            # 审校员的意见
    score: int             # 审校打分 (0-10)
    revision_count: int    # 已修改次数


# ===== 2. 三个 Agent 节点 =====
def researcher_node(state: TeamState, retriever: Any, llm: ChatOpenAI) -> dict[str, Any]:
    """研究 Agent: 从知识库检索与任务相关的信息。"""
    task = state["task"]
    print("  [研究Agent] 正在检索知识库...")

    docs = retriever.invoke(task)
    context = "\n".join(d.page_content for d in docs) if docs else "（未检索到相关信息）"

    # 让 LLM 整理检索到的信息
    msg = llm.invoke([
        SystemMessage(content="你是研究助手。请根据检索到的资料，整理出与任务相关的关键信息，简洁列出要点。"),
        HumanMessage(content=f"任务: {task}\n\n检索到的资料:\n{context}\n\n请整理关键信息:"),
    ])
    print(f"  [研究Agent] 整理完毕，{len(docs)} 条参考")
    return {"research": msg.content}


def writer_node(state: TeamState, llm: ChatOpenAI) -> dict[str, Any]:
    """写作 Agent: 基于研究结果撰写内容，若有审校意见则据此修改。"""
    task = state["task"]
    research = state["research"]
    review = state.get("review", "")
    revision_count = state.get("revision_count", 0)

    if revision_count == 0:
        print("  [写作Agent] 撰写初稿...")
        instruction = "请根据以下研究资料，完成用户的写作任务。内容应准确、结构清晰。"
    else:
        print(f"  [写作Agent] 根据审校意见进行第 {revision_count} 次修改...")
        instruction = "请根据审校意见修改你的文章，解决指出的问题。"

    msg = llm.invoke([
        SystemMessage(content=f"你是专业写手。{instruction}"),
        HumanMessage(content=(
            f"任务: {task}\n\n研究资料:\n{research}\n"
            + (f"\n审校意见:\n{review}\n" if review else "")
            + "\n请输出文章:"
        )),
    ])
    return {"draft": msg.content, "revision_count": revision_count + 1}


def reviewer_node(state: TeamState, llm: ChatOpenAI) -> dict[str, Any]:
    """审校 Agent: 审查草稿，给出意见和 0-10 分。"""
    task = state["task"]
    draft = state["draft"]

    print("  [审校Agent] 正在审校...")
    msg = llm.invoke([
        SystemMessage(content=(
            "你是严格的内容审校员。请审查文章是否准确回应了任务，"
            "内容是否完整、逻辑是否清晰。"
            "请在最后一行单独输出分数（0-10的整数），格式: 分数: X"
        )),
        HumanMessage(content=f"任务: {task}\n\n文章:\n{draft}\n\n请给出审校意见和分数:"),
    ])

    # 从回复中提取分数
    score = 5
    for line in msg.content.strip().split("\n"):
        line = line.strip()
        if "分数" in line and any(c.isdigit() for c in line):
            digits = "".join(c for c in line if c.isdigit())
            if digits:
                score = int(digits[:2]) if len(digits) >= 2 else int(digits)
                score = min(score, 10)
                break

    print(f"  [审校Agent] 打分: {score}/10")
    return {"review": msg.content, "score": score}


# ===== 3. 路由 =====
def decide_after_review(state: TeamState) -> str:
    """分数 >= 7 则通过，否则退回修改（次数用完也通过）。"""
    if state["score"] >= 7:
        print("  [路由] 审校通过！")
        return "end"
    if state["revision_count"] >= MAX_REVISIONS:
        print(f"  [路由] 已达最大修改次数({MAX_REVISIONS})，结束")
        return "end"
    print("  [路由] 审校未通过，退回修改")
    return "revise"


# ===== 4. 构建图 =====
def build_team_graph(retriever: Any, llm: ChatOpenAI):
    graph = StateGraph(TeamState)

    graph.add_node("researcher", lambda s: researcher_node(s, retriever, llm))
    graph.add_node("writer", lambda s: writer_node(s, llm))
    graph.add_node("reviewer", lambda s: reviewer_node(s, llm))

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    # 注意: reviewer 之后只能走条件边，不能加无条件边到 writer，
    # 否则条件边返回 "end" 时仍会继续走到 writer 造成死循环。

    graph.add_conditional_edges(
        "reviewer",
        decide_after_review,
        {"end": END, "revise": "writer"},
    )

    return graph.compile()


# ===== 5. 主逻辑 =====
def main():
    print("多智能体协作团队启动中...")
    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm(temperature=0.3)
    team = build_team_graph(retriever, llm)

    print("\n" + "=" * 60)
    print("  内容创作团队已就绪（输入 quit 退出）")
    print("  团队成员: 研究Agent → 写作Agent → 审校Agent")
    print("  试试: '写一篇关于RAG的短文' / '介绍LangChain的核心组件'")
    print("=" * 60 + "\n")

    while True:
        task = input("任务: ").strip()
        if task.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not task:
            continue

        print()
        result = team.invoke({
            "task": task, "research": "", "draft": "",
            "review": "", "score": 0, "revision_count": 0,
        })

        print("\n" + "=" * 60)
        print("  最终成果 (审校分数: %d/10)" % result["score"])
        print("=" * 60)
        print(result["draft"])
        print()


if __name__ == "__main__":
    main()
