import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import gzip
import json
from urllib.parse import urlencode
# 1. 读取天气服务的配置
def load_weather_config():
    # getenv读取环境变量；默认空字符串可避免变量不存在时得到None
    api_key = os.getenv("QWEATHER_API_KEY", "").strip()
    api_host = os.getenv("QWEATHER_API_HOST", "").strip()

    # 缺少任意一项就停止，并给出明确提示
    if not api_key or not api_host:
        raise ValueError("请先配置天气 API KEY 和 API Host。")

    # 返回两项配置，不打印它们
    return api_key, api_host
# 2. 将接口返回的字节数据转换为Python数据
def parse_weather_response(raw_data, content_encoding):
    # 如果响应采用gzip压缩，先解压
    if content_encoding.strip().lower() == "gzip":
        raw_data = gzip.decompress(raw_data)

    # 字节转换成文字，再将JSON文字转换为Python数据
    text = raw_data.decode("utf-8")
    return json.loads(text)
# 3. 发送一次请求，并解析天气响应
def fetch_weather(request):
    try:
        # urlopen才会真正联网；timeout避免一直等待
        # with会在读取结束后自动关闭连接
        with urlopen(request, timeout=15) as response:
            raw_data = response.read()

            # 没有压缩标记时，默认使用空字符串
            encoding = response.headers.get("Content-Encoding", "")

        return parse_weather_response(raw_data, encoding)

    # HTTPError属于URLError的一种，因此必须先处理
    except HTTPError as error:
        print("天气服务请求失败，HTTP状态码：", error.code)

    except (URLError, TimeoutError):
        print("天气请求未完成，请检查网络或稍后再试。")

    except (OSError, EOFError, UnicodeError, ValueError):
        print("天气响应解析失败，暂时无法读取结果。")

    # 失败时返回None，表示没有得到可用结果
    return None
# 检查经纬度：顺序必须是经度、纬度
def validate_coordinates(location):
    # 按英文逗号拆分，应该恰好得到两部分
    parts = location.split(",")
    if len(parts) != 2:
        raise ValueError("经纬度格式应为：116.41,39.92")

    # 将文字转换成小数；转换失败时给出易懂的提示
    try:
        longitude = float(parts[0])
        latitude = float(parts[1])
    except ValueError:
        raise ValueError("经度和纬度必须是数字。") from None

    # 经度范围为-180到180，纬度范围为-90到90
    if not (-180 <= longitude <= 180):
        raise ValueError("经度必须在-180到180之间。")

    if not (-90 <= latitude <= 90):
        raise ValueError("纬度必须在-90到90之间。")
# 构造新版实时天气地址，本函数不发送请求
def build_current_weather_url(api_host, location):
    # 输入仍使用“经度,纬度”，先统一逗号并校验范围
    location = location.replace("，", ",")
    validate_coordinates(location)

    longitude_text, latitude_text = location.split(",")

    # 转换成数值，避免把空格等原始输入直接拼入路径
    longitude = float(longitude_text)
    latitude = float(latitude_text)

    # 新版路径顺序为“纬度/经度”，坐标保留两位小数
    return (
        f"https://{api_host}/weather/v1/current/"
        f"{latitude:.2f}/{longitude:.2f}"
    )
# 展示新版天气响应，使用前先检查数据结构
def show_current_weather(result):
    if not isinstance(result, dict):
        print("天气响应格式异常：预期为JSON对象。")
        return

    condition = result.get("condition")
    temperature = result.get("temperature")
    metadata = result.get("metadata")

    # 新版把天气现象和温度分别放在两个字典中
    if not isinstance(condition, dict) or not isinstance(temperature, dict):
        print("天气响应缺少有效的天气或温度数据。")
        return

    # 数据来源声明需要随天气数据一起展示
    if not isinstance(metadata, dict):
        print("天气响应缺少来源信息。")
        return

    attributions = metadata.get("attributions")
    if not isinstance(attributions, list) or not attributions:
        print("天气响应缺少来源声明。")
        return

    print("天气：", condition.get("text", "缺失"))
    print("温度：", temperature.get("value", "缺失"),
          temperature.get("unit", ""))
    print("数据来源：和风天气")

    for attribution in attributions:
        print("来源声明：", attribution)
# 构造城市搜索地址，不发送请求
def build_city_lookup_url(api_host, city_name):
    city_name = city_name.strip()
    if not city_name:
        raise ValueError("城市名称不能为空。")

    # 编码中文名称，最多返回5个候选地点
    params = urlencode({
        "location": city_name,
        "number": 5,
    })

    return f"https://{api_host}/geo/v2/city/lookup?{params}"
# 搜索城市，返回候选地点列表；失败时返回空列表
def lookup_city(api_key, api_host, city_name):
    url = build_city_lookup_url(api_host, city_name)

    # 与天气查询使用相同的身份验证方式
    request = Request(
        url,
        headers={"X-QW-Api-Key": api_key},
        method="GET",
    )

    # 复用已有的发送、解压和JSON解析逻辑
    result = fetch_weather(request)
    if result is None:
        return []

    # 城市搜索仍使用code表示业务状态
    if not isinstance(result, dict):
        print("城市搜索响应格式异常。")
        return []

    if result.get("code") != "200":
        print("城市搜索未成功，状态码：", result.get("code", "缺失"))
        return []

    locations = result.get("location")
    if not isinstance(locations, list):
        print("城市搜索缺少有效的候选列表。")
        return []

    return locations
# 将一个候选地点转换为“经度,纬度”
def city_to_coordinates(city):
    if not isinstance(city, dict):
        raise ValueError("候选地点格式错误。")

    longitude = city.get("lon")
    latitude = city.get("lat")

    # 接口中的坐标应为非空字符串
    if not isinstance(longitude, str) or not isinstance(latitude, str):
        raise ValueError("候选地点缺少有效坐标。")

    location = f"{longitude.strip()},{latitude.strip()}"

    # 复用已有校验，确认数字格式及坐标范围
    validate_coordinates(location)
    return location
# 展示候选地点，暂时不接收选择
def show_city_candidates(cities):
    if not cities:
        print("没有找到候选地点。")
        return

    # 从1开始编号，方便用户阅读和选择
    for number, city in enumerate(cities, start=1):
        print(
            number,
            city.get("country", ""),
            city.get("adm1", ""),
            city.get("adm2", ""),
            city.get("name", "未知地点"),
        )
# 选择候选地点，取消或没有候选时返回None
def choose_city(cities):
    if not cities:
        print("没有可供选择的地点。")
        return None

    # 展示前检查候选格式，避免对错误类型调用get
    for city in cities:
        if not isinstance(city, dict):
            raise ValueError("候选地点格式错误。")

    show_city_candidates(cities)

    while True:
        choice = input("请选择地点编号（输入“退出”取消）：").strip()

        if choice == "退出":
            print("已取消地点选择。")
            return None

        try:
            number = int(choice)
        except ValueError:
            print("请输入整数编号。")
            continue

        if not 1 <= number <= len(cities):
            print("编号超出范围，请重新输入。")
            continue

        # 用户看到的编号从1开始，Python列表下标从0开始
        return cities[number - 1]
# 4. 天气查询主流程，供独立运行或统一菜单调用
def main():
    try:
        # 读取配置，并分别接收函数返回的两个值
        api_key, api_host = load_weather_config()

        # 接收地点，去掉首尾空白
        location = input(
            "请输入城市名称或经度,纬度（输入“退出”返回）："
        ).strip()

        # 在发送请求前判断，取消查询不消耗API用量
        if location == "退出":
            print("已取消天气查询。")
            return

        # 空输入没有查询意义，交给下面的except显示提示
        if not location:
            raise ValueError("查询地点不能为空。")
        # 直接输入坐标时，用坐标作为查询地点说明
        location_label = location
        # 不含逗号时，先搜索城市；坐标输入继续走原来的流程
        if "," not in location and "，" not in location:
            cities = lookup_city(api_key, api_host, location)

            # 没有候选时结束，不继续请求天气
            if not cities:
                print("未获得可用地点，本次查询结束。")
                return

            selected = choose_city(cities)

            # 用户取消选择后，不继续请求天气
            if selected is None:
                return

            # 将选中的地点转换为天气接口需要的坐标
            location = city_to_coordinates(selected)
            # 城市搜索时，记录用户最终选中的地区
            location_label = (
                f"{selected.get('country', '')} "
                f"{selected.get('adm1', '')} "
                f"{selected.get('adm2', '')} "
                f"{selected.get('name', '未知地点')}"
            ).strip()

        # 复用地址构造函数：内部完成坐标校验及顺序转换
        url = build_current_weather_url(api_host, location)
        # 请求头：随请求携带的附加信息，用于身份验证
        headers = {
            "X-QW-Api-Key": api_key,
        }

        # 创建请求对象，装入地址和请求头；此时还没有联网
        request = Request(url, headers=headers, method="GET")

        # 调用查询函数：运行到这里才会真正发送请求
        result = fetch_weather(request)

        # 新版成功响应不再依赖旧版业务字段code和now
        if result is not None:
            print("本次查询地点：", location_label)
            show_current_weather(result)
    except ValueError as error:
        print(error)



# 5. 直接运行本文件时启动；被其他模块导入时不自动查询
if __name__ == "__main__":
    main()