import os
from weather_tools import WEATHER_TOOL
import json
from weather_tools import execute_weather_tool_json

# 请求模型决定：直接回复，或者提出工具调用
# allow_tools控制本次请求是否向模型提供工具
def request_agent_reply(client, messages, allow_tools=True):
    # 第二次请求不提供工具，只让模型根据结果回答
    tool_options = {}
    if allow_tools:
        tool_options["tools"] = [WEATHER_TOOL]

    response = client.chat.completions.create(
        model=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-32B"),
        messages=messages,
        max_tokens=512,
        extra_body={"enable_thinking": False},
        **tool_options,
    )

    return response.choices[0].message

# 执行工具调用，将结果或可预期的错误整理为消息
def execute_tool_call(tool_call):
    tool_name = tool_call.function.name
    arguments_json = tool_call.function.arguments

    try:
        result = execute_weather_tool_json(
            tool_name, arguments_json
        )

        if result is None:
            result = {"error": "天气查询失败，未取得可用数据。"}

    except ValueError as error:
        # 名称、JSON或坐标不合法时，把提示返回给模型
        result = {"error": str(error)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result, ensure_ascii=False),
    }
# 处理一轮对话：最多执行一次工具、请求模型两次
def run_agent_turn(client, messages):
    # 复制列表，避免执行到一半失败时修改调用方的历史
    history = list(messages)

    first_reply = request_agent_reply(client, history)
    tool_calls = first_reply.tool_calls or []

    # 模型没有提出工具调用，直接返回文字
    if not tool_calls:
        if not first_reply.content:
            raise ValueError("模型未返回文字或工具调用。")
        return first_reply.content

    # 当前版本只支持一条调用，避免多次执行和费用失控
    if len(tool_calls) != 1:
        raise ValueError("本轮只支持一次工具调用，请缩小问题范围。")

    # 先保存模型提出调用的消息，再添加对应的工具结果
    history.append(
        first_reply.model_dump(
            include={"role", "content", "tool_calls"},
            exclude_none=True,
        )
    )
    # 开发时观察实际执行的工具，不打印参数或密钥
    print("正在执行工具：", tool_calls[0].function.name)
    history.append(execute_tool_call(tool_calls[0]))

    # 将结果交回模型，本次不再提供工具
    final_reply = request_agent_reply(
        client, history, allow_tools=False
    )

    if final_reply.tool_calls or not final_reply.content:
        raise ValueError("模型未返回有效的最终文字回复。")

    return final_reply.content
# 独立单轮入口，暂不影响原来的菜单和多轮聊天
def main():
    # 只有启动入口时才需要真实SDK
    from openai import OpenAI, APIError

    try:
        question = input("请输入问题（输入“退出”取消）：").strip()

        if question == "退出":
            print("已取消。")
            return

        if not question:
            print("问题不能为空。")
            return

        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("请先配置SILICONFLOW_API_KEY。")

        # 沿用原聊天模块的连接配置，不自动重试
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1",
            timeout=60.0,
            max_retries=0,
            default_headers={"Accept-Encoding": "identity"},
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是旅游助手，回复简洁。"
                    "查询当前天气时使用工具，不编造实时天气。"
                    "只有用户明确提供经纬度时才能查询，不得猜测坐标。"
                    "缺少坐标时请用户补充，目前不能自动搜索城市。"
                    "工具失败时如实说明，不把错误当成天气。"
                    "使用天气数据回答时，保留其来源声明链接。"
                ),
            },
            {"role": "user", "content": question},
        ]

        answer = run_agent_turn(client, messages)
        print("助手：", answer)

    except APIError:
        print("模型请求失败，请检查网络、权限、余额或模型支持情况。")
    except (ValueError, RuntimeError) as error:
        print("运行提示：", error)


# 导入本模块时不启动对话
if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n已结束。")