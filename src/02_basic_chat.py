"""
02 - LangChain 基础对话：带历史记忆的多轮对话

运行:  python 02_basic_chat.py
演示如何用 ChatPromptTemplate + 历史消息实现多轮对话。
手动维护对话历史，逻辑清晰，适合新手理解。
"""
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import get_llm


def main():
    llm = get_llm(temperature=0.7)

    # 对话模板：系统人设 + 历史 + 当前输入
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个友好的中文助手，请简洁地回答用户的问题。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm

    history = []  # 存储对话历史
    print("=" * 55)
    print("  基础对话已启动（输入 quit 退出）")
    print("=" * 55 + "\n")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not user_input:
            continue

        # 把历史和当前输入一起传给模型
        response = chain.invoke({"input": user_input, "history": history})

        # 更新历史
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response.content))

        print(f"AI: {response.content}\n")


if __name__ == "__main__":
    main()
