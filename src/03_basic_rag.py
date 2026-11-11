"""
03 - 基础 RAG：从本地知识库检索并回答

运行:  python 03_basic_rag.py
流程: 加载文档 -> 切分 -> 向量化存入 Chroma -> 检索 -> 生成
首次运行会自动生成示例文档，你也可以把 .txt / .pdf 放进 data/ 目录。
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import get_llm
from rag_utils import build_or_load_vectorstore, format_docs


def main():
    # 1. 构建/加载向量库与检索器
    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. 对话模型（RAG 推荐低 temperature）
    llm = get_llm(temperature=0)

    # 3. RAG 提示模板
    prompt = ChatPromptTemplate.from_template(
        "请根据以下检索到的上下文回答问题。如果上下文中没有答案，请说明你不知道。\n\n"
        "上下文:\n{context}\n\n问题: {question}\n回答:"
    )

    # 4. 组装 RAG 链:
    #    question -> 并行取 {context(检索), question(原样)} -> prompt -> llm -> 字符串
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n" + "=" * 55)
    print("  基础 RAG 已启动（输入 quit 退出）")
    print("=" * 55 + "\n")

    while True:
        question = input("问: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not question:
            continue

        # 先展示检索到的文档，方便理解 RAG 流程
        sources = retriever.invoke(question)
        print(f"[检索到 {len(sources)} 段相关文档]")

        answer = rag_chain.invoke(question)
        print(f"答: {answer}\n")


if __name__ == "__main__":
    main()
