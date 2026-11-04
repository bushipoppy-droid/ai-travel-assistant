import weather_service
import json
import math
# 天气工具说明，之后发送给模型，让它决定是否提出调用请求
WEATHER_TOOL = {
    "type": "function",
    "function": {
        # 名称用于对应程序中的工具函数
        "name": "get_current_weather",

        # 明确用途和限制，避免模型编造坐标
        "description": (
            "查询指定经纬度的当前天气。"
            "仅在已有明确坐标时使用，不得猜测城市坐标。"
            "不用于查询未来天气。"
        ),

        # 描述函数参数的格式
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "英文逗号分隔的经度,纬度，"
                        "例如116.41,39.92。"
                    ),
                },
            },
            "required": ["location"],
        },
    },
}

# 校验工具名称和参数，再调用允许执行的函数
def execute_weather_tool(tool_name, arguments):
    # 只允许调用明确支持的工具
    if tool_name != "get_current_weather":
        raise ValueError("不支持的工具名称。")

    # 参数必须是字典
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是字典。")

    # 必须且只能提供location，拒绝缺少或多余参数
    if set(arguments) != {"location"}:
        raise ValueError("工具参数必须且只能包含location。")

    # 先取得接口原始数据
    result = weather_service.get_current_weather(
        arguments["location"]
    )

    # 查询失败时保留None，交给Agent层转成错误消息
    if result is None:
        return None

    # 检查响应并整理，只返回必要字段
    return normalize_weather_result(result)

# 将工具参数JSON解析成字典，再交给执行入口检查
def execute_weather_tool_json(tool_name, arguments_json):
    if not isinstance(arguments_json, str):
        raise ValueError("工具参数必须是JSON字符串。")

    try:
        arguments = json.loads(arguments_json)
    except ValueError:
        raise ValueError("工具参数不是合法的JSON。") from None

    return execute_weather_tool(tool_name, arguments)
# 校验接口结果，只保留交给模型使用的必要字段
def normalize_weather_result(result):
    if not isinstance(result, dict):
        raise ValueError("天气响应必须是字典。")

    condition = result.get("condition")
    temperature = result.get("temperature")
    metadata = result.get("metadata")

    # 先确认三个字段都是字典，再读取内部数据
    for item in [condition, temperature, metadata]:
        if not isinstance(item, dict):
            raise ValueError("天气响应缺少有效的必需字段。")

    text = condition.get("text")
    value = temperature.get("value")
    unit = temperature.get("unit")
    sources = metadata.get("attributions")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("天气描述缺失。")

    # 排除布尔值、非数字、NaN和无穷大
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("温度必须是有限数字。")

    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("温度单位缺失。")

    if not isinstance(sources, list) or not sources:
        raise ValueError("天气来源声明缺失。")

    for source in sources:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("天气来源声明格式错误。")

    # 新建结果字典，不把风向等其他字段交给模型
    return {
        "weather": text,
        "temperature": value,
        "unit": unit,
        "sources": list(sources),
    }