"""
01 - 测试与 LMStudio 本地大模型的连接

运行:  python 01_test_connection.py
验证对话模型与 embedding 模型都能正常工作。
"""
import sys

from config import BASE_URL, LLM_MODEL, EMBEDDING_MODEL, get_llm, get_embeddings


def main():
    print("=" * 55)
    print("  LMStudio 连接测试")
    print("=" * 55)
    print(f"服务地址       : {BASE_URL}")
    print(f"对话模型       : {LLM_MODEL}")
    print(f"Embedding 模型 : {EMBEDDING_MODEL}")
    print("-" * 55)

    # 1. 测试对话模型
    print("\n[1/2] 测试对话模型...")
    try:
        llm = get_llm(temperature=0)
        resp = llm.invoke("你好，请用一句话介绍你自己。")
        print("  回复:", resp.content)
        print("  状态: OK")
    except Exception as e:
        print(f"  失败: {e}")
        print("\n请确认 LMStudio 已启动 Developer Server，并加载了对话模型。")
        sys.exit(1)

    # 2. 测试 embedding 模型
    print("\n[2/2] 测试 Embedding 模型...")
    try:
        embeddings = get_embeddings()
        vec = embeddings.embed_query("RAG 是检索增强生成的缩写")
        print(f"  向量维度: {len(vec)}")
        print(f"  前 5 个值: {[round(v, 4) for v in vec[:5]]}")
        print("  状态: OK")
    except Exception as e:
        print(f"  失败: {e}")
        print("\n请确认 LMStudio 已加载一个 embedding 模型（如 embeddinggemma-300m）。")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  所有测试通过，环境就绪！")
    print("=" * 55)


if __name__ == "__main__":
    main()
