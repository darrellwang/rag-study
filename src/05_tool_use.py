"""
05 - LangChain 工具调用 (Tool Calling / Function Calling)

运行:  python 05_tool_use.py

场景: 让大模型自主决定何时调用外部工具。
      这是构建 Agent 的基础——模型不再只是"说话"，而是能"做事"。

工作流:
    用户提问
        │
        ▼
    LLM 分析问题 ──► 需要工具？──否──► 直接回答
        │
       是│
        ▼
    选择工具 + 生成参数
        │
        ▼
    执行工具（本地函数）
        │
        ▼
    将工具结果交给 LLM ──► 生成最终回答

本例注册了 4 个工具: 加减乘除计算器、获取当前时间、掷骰子、模拟天气查询。
模型会根据问题自动选择合适的工具并生成调用参数。
"""
import random
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from config import get_llm


# ===== 1. 用 @tool 装饰器定义工具 =====
# docstring 会被发送给模型，模型据此判断何时使用该工具
@tool
def calculate(expression: str) -> str:
    """计算数学表达式，支持加减乘除和括号，如 '2*(3+4)'。当用户需要数学计算时使用。"""
    # 安全地计算表达式（仅允许数字和运算符）
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return f"错误: 表达式包含非法字符: {expression}"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败: {e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间。当用户询问现在几点、今天日期时使用。"""
    now = datetime.now()
    return now.strftime("现在是 %Y年%m月%d日 %H:%M:%S，星期") + \
        "一二三四五六日"[now.weekday()]


@tool
def roll_dice(sides: int = 6) -> str:
    """掷骰子，返回1到sides之间的随机数，默认6面骰。当用户想掷骰子时使用。"""
    result = random.randint(1, sides)
    return f"掷出 {sides} 面骰子，结果: {result}"


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。当用户询问天气时使用。"""
    # 模拟天气数据（实际场景中可调用天气 API）
    weathers = ["晴天", "多云", "小雨", "大雨", "雷阵雨", "小雪", "雾霾"]
    temps = range(-5, 38)
    w = random.choice(weathers)
    t = random.choice(temps)
    return f"{city}: {w}，气温 {t}°C"


# ===== 2. 主逻辑 =====
def main():
    llm = get_llm(temperature=0)
    tools = [calculate, get_current_time, roll_dice, get_weather]

    # 将工具绑定到模型，模型就能在回复中发起工具调用
    llm_with_tools = llm.bind_tools(tools)

    print("=" * 60)
    print("  工具调用演示已启动（输入 quit 退出）")
    print("=" * 60)
    print("  可用工具: calculate / get_current_time / roll_dice / get_weather")
    print("  试试问: '帮我算 123*456' / '现在几点' / '北京天气' / '掷个20面骰子'")
    print("=" * 60 + "\n")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not user_input:
            continue

        # 第一步：模型决定是否调用工具
        ai_msg = llm_with_tools.invoke([HumanMessage(content=user_input)])

        # 第二步：如果模型发起了工具调用，逐个执行
        if ai_msg.tool_calls:
            from langchain_core.messages import ToolMessage

            messages = [HumanMessage(content=user_input), ai_msg]
            for tc in ai_msg.tool_calls:
                # 找到对应的工具函数并执行
                tool_fn = {t.name: t for t in tools}[tc["name"]]
                print(f"  [调用工具] {tc['name']}({tc['args']})")
                result = tool_fn.invoke(tc["args"])
                print(f"  [工具结果] {result}")
                # 把工具结果加入对话历史
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

            # 第三步：模型基于工具结果生成最终回答
            final = llm.invoke(messages)
            print(f"AI: {final.content}\n")
        else:
            # 无需工具，直接回答
            print(f"AI: {ai_msg.content}\n")


if __name__ == "__main__":
    main()
