import os
from weather_tools import WEATHER_TOOL, CITY_SEARCH_TOOL
import json
from weather_tools import execute_weather_tool_json
import weather_service
# 请求模型决定：直接回复，或者提出工具调用
# allow_tools控制本次请求是否向模型提供工具
def request_agent_reply(client, messages, allow_tools=True):
    # 第二次请求不提供工具，只让模型根据结果回答
    tool_options = {}
    if allow_tools:
        tool_options["tools"] = [
            WEATHER_TOOL,
            CITY_SEARCH_TOOL,
        ]

    response = client.chat.completions.create(
        model=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-32B"),
        messages=messages,
        max_tokens=512,
        extra_body={"enable_thinking": False},
        **tool_options,
    )

    return response.choices[0].message

# 执行工具调用，将结果或可预期的错误整理为消息
def execute_tool_call(tool_call, candidates=None):
    tool_name = tool_call.function.name
    arguments_json = tool_call.function.arguments

    try:
        # 有候选约束时，先检查坐标，再允许实际查询
        if tool_name == "get_current_weather" and candidates is not None:
            if not isinstance(arguments_json, str):
                raise ValueError("工具参数必须是JSON字符串。")

            arguments = json.loads(arguments_json)
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是字典。")

            validate_candidate_location(
                arguments.get("location"), candidates
            )

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
# 从历史中找到最近一次城市搜索结果
def get_latest_city_candidates(messages):
    # 记录调用编号对应的工具名称，避免混淆不同工具的结果
    tool_names = {}

    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                tool_names[call["id"]] = call["function"]["name"]

    # 从后向前查找，优先使用最近的搜索结果
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue

        call_id = message.get("tool_call_id")
        if tool_names.get(call_id) != "search_city_candidates":
            continue

        try:
            result = json.loads(message["content"])
        except (ValueError, TypeError, KeyError):
            raise ValueError("历史中的城市搜索结果无法解析。") from None

        if not isinstance(result, dict):
            raise ValueError("历史中的城市搜索结果格式错误。")

        # 最近一次搜索失败时，不退回更早的候选
        if "error" in result:
            return []

        candidates = result.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("历史中的候选列表格式错误。")

        return candidates

    # None表示没有搜索历史；空列表表示搜索过但没有可用候选
    return None
# 检查天气调用坐标是否来自给定候选列表
def validate_candidate_location(location, candidates):
    # 将坐标文字转换为数字对，统一空格和逗号差异
    def coordinate_pair(value):
        if not isinstance(value, str):
            raise ValueError("候选匹配坐标必须是字符串。")

        value = value.replace("，", ",").strip()
        weather_service.validate_coordinates(value)

        longitude, latitude = value.split(",")
        return float(longitude), float(latitude)

    requested = coordinate_pair(location)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("历史候选地点格式错误。")

        candidate_location = coordinate_pair(
            candidate.get("location")
        )

        if requested == candidate_location:
            return

    raise ValueError("查询坐标不属于候选列表，请确认地点。")
# 识别“选第1个”这样的明确命令，返回对应候选
def select_candidate_by_command(question, candidates):
    question = question.strip()

    # 暂时只识别这一种格式；其他文字返回None
    if not (
        question.startswith("选第")
        and question.endswith("个")
    ):
        return None

    # 例如“选第2个”，取出中间的“2”
    number_text = question[2:-1]

    if not (number_text.isascii() and number_text.isdigit()):
        raise ValueError("请使用选第1个这样的格式。")

    if not isinstance(candidates, list) or not candidates:
        raise ValueError("没有可供选择的候选，请先搜索城市。")

    number = int(number_text)
    if not 1 <= number <= len(candidates):
        raise ValueError("候选编号超出范围。")

    # 用户编号从1开始，列表下标从0开始
    selected = candidates[number - 1]

    if not isinstance(selected, dict):
        raise ValueError("候选地点格式错误。")

    # 复用已有函数检查选中候选的坐标
    validate_candidate_location(
        selected.get("location"), [selected]
    )

    return dict(selected)
# 按候选原始顺序生成展示文字，编号由程序决定
def format_city_candidates(candidates):
    if not isinstance(candidates, list):
        raise ValueError("城市候选必须是列表。")

    if not candidates:
        return "没有可供选择的地点，请重新搜索。"

    lines = ["城市候选（以下编号由程序生成）："]

    for number, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError("候选地点格式错误。")

        # 检查候选坐标及编号；名称已由候选整理函数校验
        selected = select_candidate_by_command(
            f"选第{number}个", candidates
        )

        # 组合行政区划，省略缺失的部分
        parts = [
            selected.get("country", ""),
            selected.get("adm1", ""),
            selected.get("adm2", ""),
            selected.get("name", ""),
        ]
        label = " / ".join(part for part in parts if part)

        lines.append(
            f"{number}. {label}；坐标：{selected['location']}"
        )

    lines.append("请输入“选第1个”等命令确认地点。")
    return "\n".join(lines)
# 处理一轮对话：最多执行一次工具、请求模型两次
def run_agent_turn(client, messages):
    # 复制列表，避免执行到一半失败时修改调用方的历史
    history = list(messages)

    # 请求模型前，先由程序解析明确的候选编号
    latest_candidates = get_latest_city_candidates(history)
    question = messages[-1]["content"]
    selected = select_candidate_by_command(
        question, latest_candidates
    )

    request_messages = history

    if selected is not None:
        # 本轮确认规则只用于当前请求，不写入保存的历史
        request_messages = history + [{
            "role": "system",
            "content": (
                "本轮用户已通过编号确认地点。"
                f"查询当前天气时只能使用坐标：{selected['location']}。"
                "请调用天气工具，不得改用其他候选坐标。"
            ),
        }]

    first_reply = request_agent_reply(client, request_messages)
    tool_calls = first_reply.tool_calls or []

    # 模型没有提出工具调用，直接返回文字
    if not tool_calls:
        if not first_reply.content:
            raise ValueError("模型未返回文字或工具调用。")
        # 普通回复成功后，保存助手消息并写回历史
        history.append({
            "role": "assistant",
            "content": first_reply.content,
        })
        messages[:] = history
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
    tool_call = tool_calls[0]
    candidates = None

    if tool_call.function.name == "get_current_weather":
        candidates = latest_candidates

        if selected is not None:
            # 明确选择后，本轮只允许这一个候选的坐标
            candidates = [selected]

        elif candidates is not None:
            # 整条输入为新坐标时，以用户明确提供的坐标作为本轮约束
            question = messages[-1]["content"].replace("，", ",").strip()

            try:
                weather_service.validate_coordinates(question)
            except ValueError:
                # 普通文字确认继续使用最近一次搜索候选
                pass
            else:
                candidates = [{"location": question}]

    # 这里只表示开始处理，参数校验失败时不会真正查询
    print("正在处理工具调用：", tool_call.function.name)

    if candidates is None:
        tool_result = execute_tool_call(tool_call)
    else:
        tool_result = execute_tool_call(
            tool_call, candidates=candidates
        )

    history.append(tool_result)

    # 城市搜索成功时，由程序生成固定编号，避免模型重排
    if tool_call.function.name == "search_city_candidates":
        payload = json.loads(tool_result["content"])

        if "candidates" in payload:
            answer = format_city_candidates(payload["candidates"])

            # 保存调用、工具结果和程序生成的候选展示
            messages[:] = history + [{
                "role": "assistant",
                "content": answer,
            }]
            return answer

    # 将结果交回模型，本次不再提供工具
    final_reply = request_agent_reply(
        client, history, allow_tools=False
    )

    if final_reply.tool_calls or not final_reply.content:
        raise ValueError("模型未返回有效的最终文字回复。")

    # 新建完整历史，不继续修改刚才传给模型的列表
    completed_history = history + [{
        "role": "assistant",
        "content": final_reply.content,
    }]

    # 全部成功后，统一写回调用方的消息列表
    messages[:] = completed_history
    return final_reply.content
# 独立多轮Agent入口，暂不影响原来的统一菜单
def main():
    # 只有启动入口时才需要真实SDK
    from openai import OpenAI, APIError

    try:
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
                    "用户明确提供经纬度时，可以调用天气工具。"
                    "只有城市名称时，调用城市搜索工具，不得猜测坐标。"
                    "搜索结果不是天气，不能据此声称已查询当前天气。"
                    "搜索后展示候选名称、行政区划和对应坐标，请用户确认。"
                    "不得擅自选择第一个候选。"
                    "确认程序编号时，请用户输入“选第1个”这样的明确命令。"
                    "候选为空时如实说明未获得可用地点。"
                    "用户确认历史候选地点后，只能使用该候选已有的坐标。"
                    "确认不明确时继续追问，不得猜测地点。"
                    "工具失败时如实说明，不把错误当成天气。"
                    "天气回答只使用工具返回的weather、temperature、"
                    "unit和sources字段。"
                    "原样保留温度数值、单位和全部来源链接。"
                    "不要补充工具未返回的风向、体感温度等实时信息。"
                ),
            },
        ]

        print("旅游Agent已启动，输入“退出”结束。")

        while True:
            question = input("\n你：").strip()

            if question == "退出":
                print("对话结束。")
                break

            if not question:
                print("请输入问题，或输入“退出”。")
                continue

            # 本轮先使用临时列表，失败时不保存本轮用户问题
            turn_messages = messages + [{
                "role": "user",
                "content": question,
            }]

            try:
                answer = run_agent_turn(client, turn_messages)
            except APIError:
                print("模型请求失败，本轮未保存，请稍后再试。")
                continue
            except (ValueError, RuntimeError) as error:
                print("本轮未保存：", error)
                continue

            # 成功后保存完整历史，下一轮继续使用
            messages = turn_messages
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
