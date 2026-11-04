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
# 城市搜索工具：返回候选地点，供用户确认
CITY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_city_candidates",
        "description": (
            "根据城市或地区名称搜索候选地点及坐标。"
            "缺少明确坐标时可以使用。"
            "结果不是天气数据；应展示候选地点让用户确认，"
            "不得擅自选择第一个候选或声称已查询天气。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": {
                    "type": "string",
                    "description": "用户提供的城市或地区名称，例如北京。",
                },
            },
            "required": ["city_name"],
        },
    },
}
# 根据工具名称分发，分别检查参数和整理结果
def execute_weather_tool(tool_name, arguments):
    # 仅允许明确登记的两个工具
    if tool_name not in {
        "get_current_weather",
        "search_city_candidates",
    }:
        raise ValueError("不支持的工具名称。")

    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是字典。")

    # 城市搜索分支
    if tool_name == "search_city_candidates":
        if set(arguments) != {"city_name"}:
            raise ValueError("城市搜索参数必须且只能包含city_name。")

        candidates = weather_service.search_city_candidates(
            arguments["city_name"]
        )

        # 返回候选列表，不在这里选择地点或继续查询天气
        return {
            "candidates": normalize_city_candidates(candidates),
        }

    # 天气查询分支：保持原有行为
    if set(arguments) != {"location"}:
        raise ValueError("工具参数必须且只能包含location。")

    result = weather_service.get_current_weather(
        arguments["location"]
    )

    if result is None:
        return None

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
        "weather": text.strip(),
        "temperature": value,
        "unit": unit.strip(),
        "sources": [source.strip() for source in sources],

    }

# 校验并整理城市候选，不自动选择地点
def normalize_city_candidates(candidates):
    if not isinstance(candidates, list):
        raise ValueError("城市候选必须是列表。")

    results = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("城市候选格式错误。")

        # 新建字典，只保留名称和行政区划
        item = {}
        for field in ["name", "country", "adm1", "adm2"]:
            value = candidate.get(field, "")

            if not isinstance(value, str):
                raise ValueError("地点名称和行政区划必须是文字。")

            item[field] = value.strip()

        if not item["name"]:
            raise ValueError("候选地点缺少名称。")

        # 复用已有函数，检查坐标字段、数字格式和范围
        item["location"] = weather_service.city_to_coordinates(
            candidate
        )

        results.append(item)

    return results