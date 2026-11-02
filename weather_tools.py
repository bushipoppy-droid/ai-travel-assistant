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
# 校验天气响应，并整理出允许交给模型的必要字段
def prepare_weather_result(result):
    if not isinstance(result, dict):
        raise ValueError("天气响应必须是字典。")

    # get在字段不存在时返回None，交给下面的检查处理
    condition = result.get("condition")
    temperature = result.get("temperature")
    metadata = result.get("metadata")

    if not isinstance(condition, dict):
        raise ValueError("天气响应缺少有效的天气信息。")

    if not isinstance(temperature, dict):
        raise ValueError("天气响应缺少有效的温度信息。")

    if not isinstance(metadata, dict):
        raise ValueError("天气响应缺少有效的来源信息。")

    text = condition.get("text")
    value = temperature.get("value")
    unit = temperature.get("unit")
    attributions = metadata.get("attributions")

    # 天气描述和温度单位必须是非空字符串
    if not isinstance(text, str) or not text.strip():
        raise ValueError("天气描述必须是非空字符串。")

    # 精确检查类型，避免把True或False当成数字
    if type(value) not in (int, float):
        raise ValueError("温度必须是数字。")

    # 浮点数还可能是NaN或无穷大，它们不能作为有效温度
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("温度必须是有限数值。")

    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("温度单位必须是非空字符串。")

    if not isinstance(attributions, list) or not attributions:
        raise ValueError("天气响应必须包含来源声明列表。")

    # 每条来源声明都必须是非空字符串
    sources = []
    for attribution in attributions:
        if not isinstance(attribution, str) or not attribution.strip():
            raise ValueError("来源声明必须是非空字符串。")
        sources.append(attribution.strip())

    # 创建新字典，只选取需要的字段，不修改原始响应
    return {
        "weather": text.strip(),
        "temperature": value,
        "unit": unit.strip(),
        "attributions": sources,
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

    # 坐标类型及范围由已有天气函数继续检查
    return weather_service.get_current_weather(
        arguments["location"]
    )

# 将工具参数JSON解析成字典，再交给执行入口检查
def execute_weather_tool_json(tool_name, arguments_json):
    if not isinstance(arguments_json, str):
        raise ValueError("工具参数必须是JSON字符串。")

    try:
        arguments = json.loads(arguments_json)
    except ValueError:
        raise ValueError("工具参数不是合法的JSON。") from None

    return execute_weather_tool(tool_name, arguments)