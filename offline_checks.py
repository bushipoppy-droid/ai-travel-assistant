"""离线检查：模拟API，不使用真实密钥，不访问网络。"""
import contextlib
import copy
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types
import tempfile
import unittest
import weather_tools
import agent_chat
from unittest.mock import MagicMock, patch


# 将openai替换成仅存在于本测试进程中的假模块，确保不发请求
fake_sdk = types.ModuleType("openai")
fake_sdk.OpenAI = None
fake_sdk.APIError = type("APIError", (Exception,), {})


def load_module(name):
    path = Path(__file__).parent / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"openai": fake_sdk}):
        spec.loader.exec_module(module)
    return module


budget = load_module("budget_agent")
chat = load_module("travel_assistant")
# 菜单导入的也必须是假客户端对应模块，不加载真实SDK
with patch.dict(sys.modules, {
    "openai": fake_sdk,
    "budget_agent": budget,
    "travel_assistant": chat,
}):
    menu = load_module("main")


class FakeClient:
    # 记录每次请求的消息快照，用于验证历史是否真的传入
    def __init__(self, text, finish_reason="stop"):
        self.text = text
        self.finish_reason = finish_reason
        self.requests = []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content=self.text), finish_reason=self.finish_reason
        )
        return types.SimpleNamespace(choices=[choice])


# 每次预算测试都重定向__file__到临时目录，绝不写入项目目录
# 临时目录退出时自动清理；真实budget_result.json保持不变
@contextlib.contextmanager
def isolated_storage():
    with tempfile.TemporaryDirectory(prefix="travel-check-") as directory:
        with patch.object(budget, "__file__", str(Path(directory) / "budget_agent.py")):
            yield Path(directory)


def run_budget(text, inputs=None, finish_reason="stop"):
    client = FakeClient(text, finish_reason)
    output = io.StringIO()
    with isolated_storage(), \
         patch.dict(os.environ, {"SILICONFLOW_API_KEY": "offline-placeholder"}), \
         patch.object(budget, "OpenAI", return_value=client), \
         patch("builtins.input", side_effect=inputs or ["深圳", "3", "1500"]), \
         contextlib.redirect_stdout(output):
        budget.main()
    return output.getvalue(), client


# 持久化测试在同一个临时目录中连续执行，验证覆盖或保留旧文件
# 仅供本文件测试使用；外层测试必须先进入isolated_storage()
def generate_in_storage(text, finish_reason="stop"):
    output = io.StringIO()
    with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "offline-placeholder"}), \
         patch.object(budget, "OpenAI", return_value=FakeClient(text, finish_reason)), \
         patch("builtins.input", side_effect=["深圳", "3", "1500"]), \
         contextlib.redirect_stdout(output):
        budget.main()
    return output.getvalue()


def read_latest_output():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        menu.show_latest_budget()
    return output.getvalue()


class OfflineChecks(unittest.TestCase):
    def test_valid_budget(self):
        output, client = run_budget('{"ticket":100,"transport":80,"food":500,"other":200}')
        self.assertIn("880", output)
        self.assertIn("620", output)
        self.assertEqual(len(client.requests), 1)
        self.assertIn("深圳", client.requests[0][-1]["content"])

    def test_over_budget_and_fence(self):
        output, _ = run_budget('```json\n{"ticket":500,"transport":200,"food":600,"other":400}\n```')
        self.assertIn("超出预算： 200", output)

    def test_invalid_json(self):
        output, _ = run_budget('以下是预算：{}')
        self.assertIn("解析错误", output)

    def test_non_object(self):
        output, _ = run_budget('[100,80,500,200]')
        self.assertIn("格式错误", output)

    def test_missing_and_extra_fields(self):
        for data in [dict(ticket=100), dict(ticket=100, transport=80, food=500, other=200, extra=1)]:
            with self.subTest(data=data):
                output, _ = run_budget(json.dumps(data))
                self.assertIn("字段错误", output)

    def test_invalid_amounts(self):
        for amount in [-1, "500元", True, 1.5]:
            with self.subTest(amount=amount):
                data = dict(ticket=100, transport=80, food=amount, other=200)
                output, _ = run_budget(json.dumps(data))
                self.assertIn("金额错误", output)
                self.assertNotIn("总费用：", output)

    def test_input_recovery(self):
        output, client = run_budget('{"ticket":100,"transport":80,"food":500,"other":200}',
                                    ["深圳", "三天", "0", "3", "abc", "-1", "1500"])
        self.assertIn("游玩天数必须大于0", output)
        self.assertIn("预算必须大于0", output)
        self.assertEqual(len(client.requests), 1)

    def test_chat_history_blank_exit(self):
        client = FakeClient("模拟回复")
        output = io.StringIO()
        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "offline-placeholder"}), \
             patch.object(chat, "OpenAI", return_value=client), \
             patch("builtins.input", side_effect=["", "杭州两天", "刚才去哪", "退出"]), \
             contextlib.redirect_stdout(output):
            chat.main()
        self.assertEqual(len(client.requests), 2)
        self.assertEqual([m["role"] for m in client.requests[1]],
                         ["system", "user", "assistant", "user"])
        self.assertIn("当前消息数量： 5", output.getvalue())
        self.assertIn("对话结束", output.getvalue())


    def test_save_latest_and_replace(self):
        with isolated_storage() as directory:
            path = directory / "budget_result.json"
            generate_in_storage('{"ticket":100,"transport":80,"food":500,"other":200}')
            first = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(first["total_cost"], 880)
            self.assertEqual(first["remaining"], 620)
            self.assertFalse(first["includes_accommodation"])
            generate_in_storage('{"ticket":500,"transport":200,"food":600,"other":400}')
            latest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(latest["total_cost"], 1700)
            self.assertEqual(latest["remaining"], -200)
            self.assertTrue(latest["over_budget"])
            self.assertFalse((directory / "budget_result.json.tmp").exists())

    def test_invalid_or_truncated_keeps_old_file(self):
        cases = [
            ("not JSON", "stop"),
            ('{"ticket":1}', "stop"),
            ('{"ticket":1,"transport":2,"food":-1,"other":3}', "stop"),
            ('{"ticket":1,"transport":2,"food":3,"other":4}', "length"),
        ]
        with isolated_storage() as directory:
            path = directory / "budget_result.json"
            path.write_text("original-budget", encoding="utf-8")
            for text, reason in cases:
                with self.subTest(text=text, reason=reason):
                    generate_in_storage(text, reason)
                    self.assertEqual(path.read_text(encoding="utf-8"), "original-budget")

    def test_save_failure_keeps_old_and_cleans_temp(self):
        with isolated_storage() as directory:
            path = directory / "budget_result.json"
            path.write_text("old", encoding="utf-8")
            with patch.object(budget.os, "replace", side_effect=OSError("simulated")):
                output = generate_in_storage('{"ticket":100,"transport":80,"food":500,"other":200}')
            self.assertIn("预算保存失败", output)
            self.assertEqual(path.read_text(encoding="utf-8"), "old")
            self.assertFalse((directory / "budget_result.json.tmp").exists())

    def test_view_missing_or_corrupt_file(self):
        with isolated_storage() as directory:
            self.assertIn("还没有保存的预算", read_latest_output())
            (directory / "budget_result.json").write_text("broken", encoding="utf-8")
            self.assertIn("预算文件读取失败", read_latest_output())

    def test_view_recalculates_totals(self):
        with isolated_storage() as directory:
            generate_in_storage('{"ticket":100,"transport":80,"food":500,"other":200}')
            path = directory / "budget_result.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["total_cost"] = 99999
            data["remaining"] = 99999
            path.write_text(json.dumps(data), encoding="utf-8")
            output = read_latest_output()
            self.assertIn("总费用： 880", output)
            self.assertIn("剩余预算： 620", output)
            self.assertNotIn("99999", output)

    def test_view_rejects_invalid_data(self):
        cases = [[], {},
                 {"destination":"深圳","days":3,"budget":1500,"costs":[]},
                 {"destination":"深圳","days":3,"budget":1500,"costs":{"ticket":1}},
                 {"destination":"深圳","days":3,"budget":1500,
                  "costs":{"ticket":1,"transport":2,"food":-1,"other":3}},
                 {"destination":"深圳","days":3,"budget":"1500",
                  "costs":{"ticket":1,"transport":2,"food":3,"other":4}}]
        with isolated_storage() as directory:
            for data in cases:
                with self.subTest(data=data):
                    (directory / "budget_result.json").write_text(json.dumps(data), encoding="utf-8")
                    self.assertNotIn("===== 最新保存的预算 =====", read_latest_output())

    def test_menu_dispatch_and_return(self):
        # 所有功能入口均使用模拟函数，不调用真实API
        output = io.StringIO()

        with (
            patch(
                "builtins.input",
                side_effect=["9", "1", "2", "3", "4", "5", "0"],
            ),
            patch.object(chat, "main") as chat_call,
            patch.object(budget, "main") as budget_call,
            patch.object(menu, "show_latest_budget") as view_call,
            patch.object(
                menu.weather_service, "main"
            ) as weather_call,
            patch.object(
                menu.agent_chat, "main"
            ) as agent_call,
            contextlib.redirect_stdout(output),
        ):
            menu.main()

        chat_call.assert_called_once()
        budget_call.assert_called_once()
        view_call.assert_called_once()
        weather_call.assert_called_once()
        agent_call.assert_called_once()

        self.assertIn("选项无效", output.getvalue())
        self.assertIn("旅游助手已关闭", output.getvalue())

    def test_menu_error_returns_to_menu(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["2", "0"]), \
             patch.object(budget, "main", side_effect=ValueError("模拟错误")), \
             contextlib.redirect_stdout(output):
            menu.main()
        self.assertIn("运行提示： 模拟错误", output.getvalue())
        self.assertIn("旅游助手已关闭", output.getvalue())

    # 检查未压缩的响应，包含中文天气描述
    def test_weather_parse_plain(self):
        expected = {"code": "200", "now": {"text": "晴", "temp": "20"}}

        # 将字典转换成JSON文字，再转换成模拟响应字节
        raw_data = json.dumps(expected, ensure_ascii=False).encode("utf-8")

        result = menu.weather_service.parse_weather_response(raw_data, "")
        self.assertEqual(result, expected)

    # 检查压缩响应是否能解压，并保留完整天气数据
    def test_weather_parse_gzip(self):
        expected = {"code": "200", "now": {"text": "多云", "temp": "18"}}
        raw_data = json.dumps(expected, ensure_ascii=False).encode("utf-8")

        compressed_data = gzip.compress(raw_data)
        result = menu.weather_service.parse_weather_response(
            compressed_data, "gzip"
        )
        self.assertEqual(result, expected)
    # 检查断网时是否显示提示，并返回None
    def test_weather_network_error(self):
        weather = menu.weather_service
        output = io.StringIO()

        # 临时替换联网函数，主动抛出模拟断网异常
        with patch.object(
            weather,
            "urlopen",
            side_effect=weather.URLError("模拟断网"),
        ) as mock_open:
            # 收集提示文字，便于后面检查
            with contextlib.redirect_stdout(output):
                result = weather.fetch_weather("offline-request")

        # 失败后应返回None，并显示网络提示
        self.assertIsNone(result)
        self.assertIn("天气请求未完成", output.getvalue())

        # 确认只尝试了一次，并且设置了超时
        mock_open.assert_called_once_with("offline-request", timeout=15)
    # 检查取消查询和非法输入：应显示提示，不能发送请求
    def test_weather_invalid_input_no_request(self):
        weather = menu.weather_service

        # 每组数据包含：模拟输入、预期提示
        cases = [
            ("退出", "已取消天气查询"),
            ("", "查询地点不能为空"),
            ("200,39.92", "经度必须在"),
            ("116.41,100", "纬度必须在"),
            ("abc,39.92", "必须是数字"),
        ]

        for location, expected_message in cases:
            with self.subTest(location=location):
                output = io.StringIO()

                # 使用假配置，避免读取真实密钥
                # 替换联网函数，即使程序出错也不会真的发送请求
                with patch.object(
                    weather, "load_weather_config",
                    return_value=("offline-placeholder", "example.invalid"),
                ), patch(
                    "builtins.input", return_value=location,
                ), patch.object(
                    weather, "urlopen",
                ) as mock_open, contextlib.redirect_stdout(output):
                    weather.main()

                self.assertIn(expected_message, output.getvalue())
                mock_open.assert_not_called()
    # 检查新版地址：用户输入经度在前，接口路径纬度在前
    def test_weather_v1_url(self):
        weather = menu.weather_service

        url = weather.build_current_weather_url(
            "example.invalid", "116.41,39.92"
        )

        self.assertEqual(
            url,
            "https://example.invalid/weather/v1/current/39.92/116.41",
        )
    # 检查新版天气数据是否正确展示
    def test_weather_v1_display(self):
        sample = {
            "condition": {"text": "晴"},
            "temperature": {"value": 20.5, "unit": "°C"},
            "metadata": {
                "attributions": [
                    "https://developer.qweather.com/attribution.html"
                ]
            },
        }
        output = io.StringIO()

        # 收集展示文字，不发送请求
        with contextlib.redirect_stdout(output):
            menu.weather_service.show_current_weather(sample)

        text = output.getvalue()
        self.assertIn("天气： 晴", text)
        self.assertIn("20.5 °C", text)
        self.assertIn(
            "https://developer.qweather.com/attribution.html", text
        )
    # 检查新版主流程：构造请求后，能展示模拟天气
    def test_weather_v1_main(self):
        weather = menu.weather_service
        output = io.StringIO()
        sample = {
            "condition": {"text": "晴"},
            "temperature": {"value": 20.5, "unit": "°C"},
            "metadata": {
                "attributions": [
                    "https://developer.qweather.com/attribution.html"
                ]
            },
        }

        # 假配置、假输入、假查询，不读取真实密钥或访问网络
        with patch.object(
            weather, "load_weather_config",
            return_value=("offline-placeholder", "example.invalid"),
        ), patch(
            "builtins.input", return_value="116.41,39.92",
        ), patch.object(
            weather, "fetch_weather", return_value=sample,
        ) as mock_fetch, contextlib.redirect_stdout(output):
            weather.main()

        mock_fetch.assert_called_once()

        # 取出传给查询函数的请求对象，检查实际生成的地址
        request = mock_fetch.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://example.invalid/weather/v1/current/39.92/116.41",
        )
        self.assertEqual(request.get_method(), "GET")

        self.assertIn("天气： 晴", output.getvalue())
        self.assertIn("20.5 °C", output.getvalue())

    # 检查编号输错后能恢复，并返回用户选中的地点
    def test_city_selection_input_recovery(self):
        cities = [
            {"name": "北京", "lon": "116.41", "lat": "39.92"},
            {"name": "鼓楼", "lon": "118.77", "lat": "32.07"},
        ]
        output = io.StringIO()

        # 依次模拟非数字、越界编号和有效编号
        with patch(
            "builtins.input", side_effect=["abc", "9", "2"],
        ), contextlib.redirect_stdout(output):
            selected = menu.weather_service.choose_city(cities)

        self.assertEqual(selected, cities[1])
        self.assertIn("请输入整数编号", output.getvalue())
        self.assertIn("编号超出范围", output.getvalue())
    # 检查用户可以取消地点选择
    def test_city_selection_cancel(self):
        cities = [
            {"name": "北京", "lon": "116.41", "lat": "39.92"},
        ]
        output = io.StringIO()

        with patch(
            "builtins.input", return_value="退出",
        ), contextlib.redirect_stdout(output):
            selected = menu.weather_service.choose_city(cities)

        self.assertIsNone(selected)
        self.assertIn("已取消地点选择", output.getvalue())
    # 检查城市搜索与天气查询的完整串联，不访问真实API
    def test_city_weather_main(self):
        weather = menu.weather_service
        output = io.StringIO()

        city_response = {
            "code": "200",
            "location": [
                {"name": "北京", "lon": "116.41", "lat": "39.92"}
            ],
        }
        weather_response = {
            "condition": {"text": "晴"},
            "temperature": {"value": 20.5, "unit": "°C"},
            "metadata": {
                "attributions": [
                    "https://developer.qweather.com/attribution.html"
                ]
            },
        }

        # 第一次模拟城市搜索响应，第二次模拟天气响应
        with patch.object(
            weather, "load_weather_config",
            return_value=("offline-placeholder", "example.invalid"),
        ), patch(
            "builtins.input", side_effect=["北京", "1"],
        ), patch.object(
            weather, "fetch_weather",
            side_effect=[city_response, weather_response],
        ) as mock_fetch, contextlib.redirect_stdout(output):
            weather.main()

        self.assertEqual(mock_fetch.call_count, 2)

        # 按调用顺序取出两个请求，确认搜索和天气地址
        search_request = mock_fetch.call_args_list[0].args[0]
        weather_request = mock_fetch.call_args_list[1].args[0]

        self.assertEqual(
            search_request.full_url,
            "https://example.invalid/geo/v2/city/lookup"
            "?location=%E5%8C%97%E4%BA%AC&number=5",
        )
        self.assertEqual(
            weather_request.full_url,
            "https://example.invalid/weather/v1/current/39.92/116.41",
        )
        self.assertIn("天气： 晴", output.getvalue())
        self.assertIn("20.5 °C", output.getvalue())
        # 确认显示的是选中的城市名称，而非搜索词或坐标
        self.assertIn("本次查询地点： 北京", output.getvalue())

    # 检查天气结果整理：保留必要字段，且不修改原始数据
    def test_normalize_weather_result_success(self):
        sample = {
            "condition": {"text": " 多云 "},
            "temperature": {"value": -2.5, "unit": " °C "},
            "metadata": {
                "attributions": [" https://example.invalid/source "]
            },
            "humidity": 82,
        }

        # 深拷贝作为处理前的快照，用于检查原始数据是否被修改
        original = copy.deepcopy(sample)

        result = weather_tools.normalize_weather_result(sample)

        # 比较整个字典，也能发现意外保留的额外字段
        self.assertEqual(result, {
            "weather": "多云",
            "temperature": -2.5,
            "unit": "°C",
            "sources": ["https://example.invalid/source"],
        })
        self.assertEqual(sample, original)
    # 检查不合格的天气数据必须被拒绝
    def test_normalize_weather_result_rejects_invalid_data(self):
        valid = {
            "condition": {"text": "晴"},
            "temperature": {"value": 20.5, "unit": "°C"},
            "metadata": {
                "attributions": ["https://example.invalid/source"]
            },
        }

        # 顶层类型错误，以及缺少必要信息
        cases = [None, [], {}]

        # 每次只替换一个部分，让错误原因更明确
        replacements = [
            ("condition", None),
            ("condition", {"text": "   "}),
            ("temperature", None),
            ("temperature", {"value": True, "unit": "°C"}),
            ("temperature", {"value": "20.5", "unit": "°C"}),
            ("temperature", {"value": float("nan"), "unit": "°C"}),
            ("temperature", {"value": float("inf"), "unit": "°C"}),
            ("temperature", {"value": 20.5, "unit": ""}),
            ("metadata", None),
            ("metadata", {"attributions": []}),
            ("metadata", {"attributions": ["   "]}),
            ("metadata", {"attributions": [123]}),
        ]

        for field, invalid_value in replacements:
            sample = copy.deepcopy(valid)
            sample[field] = invalid_value
            cases.append(sample)

        for index, sample in enumerate(cases):
            with self.subTest(case=index):
                # 必须抛出ValueError；没有抛出就判定测试失败
                with self.assertRaises(ValueError):
                    weather_tools.normalize_weather_result(sample)
    # 检查天气工具能接收坐标、构造请求并返回数据
    def test_weather_tool_success(self):
        weather = menu.weather_service
        sample = {
            "condition": {"text": "晴"},
            "temperature": {"value": 20.5, "unit": "°C"},
        }

        with (
            # 使用假配置，不读取真实密钥
            patch.object(
                weather,
                "load_weather_config",
                return_value=("fake-key", "example.invalid"),
            ),

            # 替换发送函数，不联网
            patch.object(
                weather,
                "fetch_weather",
                return_value=sample,
            ) as mock_fetch,
        ):
            result = weather.get_current_weather("116.41,39.92")

        # 查询结果应完整返回，且只请求一次
        self.assertEqual(result, sample)
        mock_fetch.assert_called_once()

        # 检查实际构造的请求地址与方法
        request = mock_fetch.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://example.invalid/weather/v1/current/39.92/116.41",
        )
        self.assertEqual(request.get_method(), "GET")
    # 检查工具参数不合法时，在读取配置和请求之前停止
    def test_weather_tool_invalid_input(self):
        weather = menu.weather_service
        cases = [
            None,
            123,
            "",
            "   ",
            "200,39.92",
            "abc,39.92",
        ]

        for location in cases:
            with self.subTest(location=location):
                with (
                    patch.object(
                        weather, "load_weather_config"
                    ) as mock_config,
                    patch.object(
                        weather, "fetch_weather"
                    ) as mock_fetch,
                ):
                    # 非法输入应主动抛出ValueError
                    with self.assertRaises(ValueError):
                        weather.get_current_weather(location)

                mock_config.assert_not_called()
                mock_fetch.assert_not_called()

    # 检查参数传递，并确认只返回整理后的天气字段
    def test_weather_tool_json_success(self):
        sample = {
            "condition": {"text": "阴"},
            "temperature": {"value": 9.06, "unit": "°C"},
            "metadata": {
                "attributions": [
                    "https://developer.qweather.com/attribution.html"
                ]
            },
            "wind": {"direction": {"compass": "sse"}},
        }

        with patch.object(
                weather_tools.weather_service,
                "get_current_weather",
                return_value=sample,
        ) as mock_weather:
            result = weather_tools.execute_weather_tool_json(
                "get_current_weather",
                '{"location": "116.41,39.92"}',
            )

        # 完整比较，确认风向等多余字段没有进入结果
        self.assertEqual(result, {
            "weather": "阴",
            "temperature": 9.06,
            "unit": "°C",
            "sources": [
                "https://developer.qweather.com/attribution.html"
            ],
        })
        mock_weather.assert_called_once_with("116.41,39.92")

    # 检查错误调用被拒绝，且没有执行天气函数
    def test_weather_tool_json_rejects_invalid_call(self):
        cases = [
            ("unknown_tool", '{"location": "116.41,39.92"}'),
            ("get_current_weather", None),
            ("get_current_weather", "not JSON"),
            ("get_current_weather", "[]"),
            ("get_current_weather", "{}"),
            (
                "get_current_weather",
                '{"location": "116.41,39.92", "extra": 1}',
            ),
        ]

        for tool_name, arguments_json in cases:
            with self.subTest(
                tool=tool_name, arguments=arguments_json
            ):
                with patch.object(
                    weather_tools.weather_service,
                    "get_current_weather",
                ) as mock_weather:
                    with self.assertRaises(ValueError):
                        weather_tools.execute_weather_tool_json(
                            tool_name, arguments_json
                        )

                mock_weather.assert_not_called()
    # 检查一轮Agent流程：模型调用工具，再根据结果回答
    def test_agent_tool_round_trip(self):
        tool_call = types.SimpleNamespace(
            id="call_test_1",
            function=types.SimpleNamespace(
                name="get_current_weather",
                arguments='{"location": "116.41,39.92"}',
            ),
        )

        # 模拟第一次助手消息及其转换成字典的结果
        first_reply = MagicMock()
        first_reply.tool_calls = [tool_call]
        first_reply.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_test_1",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": '{"location": "116.41,39.92"}',
                },
            }],
        }

        final_reply = types.SimpleNamespace(
            content="该地点当前晴，20.5℃。",
            tool_calls=None,
        )

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=first_reply)]
            ),
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=final_reply)]
            ),
        ]

        messages = [{
            "role": "user",
            "content": "查询116.41,39.92的当前天气。",
        }]

        # 模拟接口原始数据，额外包含不需要交给模型的风向
        sample = {
            "condition": {"text": "晴"},
            "temperature": {"value": 20.5, "unit": "°C"},
            "metadata": {
                "attributions": [
                    "https://developer.qweather.com/attribution.html"
                ]
            },
            "wind": {"direction": {"compass": "sse"}},
        }

        # 只替换天气查询，保留工具校验与结果整理
        with patch.object(
                weather_tools.weather_service,
                "get_current_weather",
                return_value=sample,
        ) as mock_weather:
            answer = agent_chat.run_agent_turn(client, messages)

        self.assertEqual(answer, "该地点当前晴，20.5℃。")
        mock_weather.assert_called_once_with("116.41,39.92")

        calls = client.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIn("tools", calls[0].kwargs)
        self.assertNotIn("tools", calls[1].kwargs)

        sent = calls[1].kwargs["messages"]
        self.assertEqual(
            [message["role"] for message in sent],
            ["user", "assistant", "tool"],
        )
        self.assertEqual(sent[-1]["tool_call_id"], "call_test_1")
        # 检查第二次模型请求收到的工具结果
        tool_result = json.loads(sent[-1]["content"])
        self.assertEqual(tool_result, {
            "weather": "晴",
            "temperature": 20.5,
            "unit": "°C",
            "sources": [
                "https://developer.qweather.com/attribution.html"
            ],
        })
        # 成功后应保存本轮完整历史
        self.assertEqual(
            [message["role"] for message in messages],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(
            messages[-1]["content"],
            "该地点当前晴，20.5℃。",
        )

    # 检查天气数据不完整时，转换为对应的工具错误消息
    def test_agent_invalid_weather_returns_error(self):
        tool_call = types.SimpleNamespace(
            id="call_invalid_weather",
            function=types.SimpleNamespace(
                name="get_current_weather",
                arguments='{"location": "116.41,39.92"}',
            ),
        )

        # 模拟接口返回残缺数据：有天气描述，但缺少温度和来源
        with patch.object(
            weather_tools.weather_service,
            "get_current_weather",
            return_value={"condition": {"text": "晴"}},
        ) as mock_weather:
            message = agent_chat.execute_tool_call(tool_call)

        mock_weather.assert_called_once_with("116.41,39.92")

        # 即使查询结果不合格，也必须保留对应的工具调用编号
        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], tool_call.id)

        content = json.loads(message["content"])
        self.assertEqual(content, {
            "error": "天气响应缺少有效的必需字段。"
        })
    # 普通聊天：模型直接回答，不执行工具，也不请求第二次
    def test_agent_direct_reply(self):
        reply = types.SimpleNamespace(
            content="你好，请问想去哪里旅游？",
            tool_calls=None,
        )

        with (
            patch.object(
                agent_chat, "request_agent_reply",
                return_value=reply,
            ) as mock_reply,
            patch.object(
                agent_chat, "execute_tool_call"
            ) as mock_tool,
        ):
            answer = agent_chat.run_agent_turn(
                MagicMock(),
                [{"role": "user", "content": "你好"}],
            )

        self.assertEqual(answer, "你好，请问想去哪里旅游？")
        mock_reply.assert_called_once()
        mock_tool.assert_not_called()

    # 模型一次提出多个调用：执行前拒绝，避免超出当前能力范围
    def test_agent_rejects_multiple_tool_calls(self):
        reply = types.SimpleNamespace(
            content=None,
            tool_calls=[MagicMock(), MagicMock()],
        )

        with (
            patch.object(
                agent_chat, "request_agent_reply",
                return_value=reply,
            ) as mock_reply,
            patch.object(
                agent_chat, "execute_tool_call"
            ) as mock_tool,
        ):
            with self.assertRaisesRegex(
                ValueError, "本轮只支持一次工具调用"
            ):
                agent_chat.run_agent_turn(
                    MagicMock(),
                    [{"role": "user", "content": "查询两个地点天气"}],
                )

        mock_reply.assert_called_once()
        mock_tool.assert_not_called()
    # 检查异常温度被拒绝，并转换成可回传的工具错误
    def test_agent_rejects_invalid_temperature(self):
        tool_call = types.SimpleNamespace(
            id="call_invalid_temperature",
            function=types.SimpleNamespace(
                name="get_current_weather",
                arguments='{"location": "116.41,39.92"}',
            ),
        )

        # 布尔值、文字、缺失值、非有限数字都不应作为温度
        cases = [True, "20.5", None, float("nan"), float("inf")]

        for value in cases:
            with self.subTest(temperature=value):
                sample = {
                    "condition": {"text": "晴"},
                    "temperature": {"value": value, "unit": "°C"},
                    "metadata": {
                        "attributions": [
                            "https://developer.qweather.com/attribution.html"
                        ]
                    },
                }

                # 只模拟天气查询，保留参数与响应校验的真实逻辑
                with patch.object(
                    weather_tools.weather_service,
                    "get_current_weather",
                    return_value=sample,
                ) as mock_weather:
                    message = agent_chat.execute_tool_call(tool_call)

                result = json.loads(message["content"])

                self.assertEqual(
                    result,
                    {"error": "温度必须是有限数字。"},
                )
                self.assertEqual(
                    message["tool_call_id"],
                    "call_invalid_temperature",
                )
                mock_weather.assert_called_once_with("116.41,39.92")
    # 检查城市搜索工具的参数传递、候选整理和结果回传
    def test_agent_city_search_returns_candidates(self):
        tool_call = types.SimpleNamespace(
            id="call_city_1",
            function=types.SimpleNamespace(
                name="search_city_candidates",
                arguments='{"city_name": "怀柔"}',
            ),
        )

        sample = [{
            "name": " 怀柔 ",
            "country": "中国",
            "adm1": "北京市",
            "adm2": "北京",
            "lon": "116.63",
            "lat": "40.32",
            "rank": "额外字段",
        }]

        with (
            patch.object(
                weather_tools.weather_service,
                "load_weather_config",
                return_value=("fake-key", "example.invalid"),
            ),
            patch.object(
                weather_tools.weather_service,
                "lookup_city",
                return_value=sample,
            ) as mock_lookup,
            patch.object(
                weather_tools.weather_service,
                "get_current_weather",
            ) as mock_weather,
        ):
            message = agent_chat.execute_tool_call(tool_call)

        mock_lookup.assert_called_once_with(
            "fake-key", "example.invalid", "怀柔"
        )

        # 搜索地点不等于查天气，不能偷偷继续执行
        mock_weather.assert_not_called()

        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], "call_city_1")
        self.assertEqual(json.loads(message["content"]), {
            "candidates": [{
                "name": "怀柔",
                "country": "中国",
                "adm1": "北京市",
                "adm2": "北京",
                "location": "116.63,40.32",
            }],
        })
    # 检查城市搜索结果能回传给模型，且不会继续查询天气
    def test_agent_city_search_round_trip(self):
        tool_call = types.SimpleNamespace(
            id="call_city_flow",
            function=types.SimpleNamespace(
                name="search_city_candidates",
                arguments='{"city_name": "怀柔"}',
            ),
        )

        first_reply = MagicMock()
        first_reply.tool_calls = [tool_call]
        first_reply.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [{
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }],
        }

        candidates = [{
            "name": "怀柔",
            "country": "中国",
            "adm1": "北京市",
            "adm2": "北京",
            "lon": "116.63",
            "lat": "40.32",
        }]

        messages = [{
            "role": "user",
            "content": "怀柔天气怎么样？",
        }]

        with (
            # 模拟一次模型回复，候选展示由程序生成
            patch.object(
                agent_chat,
                "request_agent_reply",
                return_value=first_reply,
            ) as mock_reply,
            patch.object(
                weather_tools.weather_service,
                "search_city_candidates",
                return_value=candidates,
            ) as mock_search,
            patch.object(
                weather_tools.weather_service,
                "get_current_weather",
            ) as mock_weather,
        ):
            answer = agent_chat.run_agent_turn(
                MagicMock(), messages
            )

        self.assertEqual(answer, (
            "城市候选（以下编号由程序生成）：\n"
            "1. 中国 / 北京市 / 北京 / 怀柔；坐标：116.63,40.32\n"
            "请输入“选第1个”等命令确认地点。"
        ))
        mock_search.assert_called_once_with("怀柔")
        mock_weather.assert_not_called()
        mock_reply.assert_called_once()

        # 完整历史仍然保存，供下一轮确认使用
        self.assertEqual(
            [message["role"] for message in messages],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(messages[-1]["content"], answer)

        tool_message = messages[-2]
        self.assertEqual(tool_message["tool_call_id"], tool_call.id)
        self.assertEqual(json.loads(tool_message["content"]), {
            "candidates": [{
                "name": "怀柔",
                "country": "中国",
                "adm1": "北京市",
                "adm2": "北京",
                "location": "116.63,40.32",
            }],
        })

    # 第二次模型请求失败时，不保存未完成的助手和工具消息
    def test_agent_failure_keeps_history(self):
        tool_data = {
            "id": "call_failure",
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "arguments": '{"location": "116.41,39.92"}',
            },
        }

        tool_call = types.SimpleNamespace(
            id=tool_data["id"],
            function=types.SimpleNamespace(
                **tool_data["function"]
            ),
        )

        first_reply = MagicMock()
        first_reply.tool_calls = [tool_call]
        first_reply.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [tool_data],
        }

        messages = [{
            "role": "user",
            "content": "查询116.41,39.92的天气。",
        }]
        original = copy.deepcopy(messages)

        with (
            # 第一次提出调用，第二次模拟请求失败
            patch.object(
                agent_chat,
                "request_agent_reply",
                side_effect=[
                    first_reply,
                    RuntimeError("模拟模型请求失败"),
                ],
            ) as mock_reply,
            patch.object(
                agent_chat,
                "execute_tool_call",
                return_value={
                    "role": "tool",
                    "tool_call_id": "call_failure",
                    "content": '{"weather": "晴"}',
                },
            ) as mock_tool,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "模拟模型请求失败"
            ):
                agent_chat.run_agent_turn(MagicMock(), messages)

        self.assertEqual(messages, original)
        self.assertEqual(mock_reply.call_count, 2)
        mock_tool.assert_called_once_with(tool_call)
    # 检查多轮循环：空输入跳过，成功保留，失败不污染历史
    def test_agent_main_keeps_only_completed_turns(self):
        received = []
        output = io.StringIO()

        # 模拟一轮处理，并记录每轮实际收到的历史
        def fake_turn(client, messages):
            received.append(copy.deepcopy(messages))
            question = messages[-1]["content"]

            if question == "模拟失败":
                # 故意加入半轮消息，检查入口是否会丢弃
                messages.append({
                    "role": "assistant",
                    "content": "未完成的回复",
                })
                raise RuntimeError("模拟失败")

            answer = "回答：" + question
            messages.append({
                "role": "assistant",
                "content": answer,
            })
            return answer

        with (
            # 替换整个SDK，创建客户端也不会联网
            patch.dict(sys.modules, {"openai": fake_sdk}),
            patch.object(fake_sdk, "OpenAI") as mock_client,
            patch.dict(
                os.environ,
                {"SILICONFLOW_API_KEY": "fake-key"},
            ),
            patch(
                "builtins.input",
                side_effect=["", "第一问", "模拟失败", "继续问", "退出"],
            ),
            patch.object(
                agent_chat,
                "run_agent_turn",
                side_effect=fake_turn,
            ) as mock_turn,
            contextlib.redirect_stdout(output),
        ):
            agent_chat.main()

        # 客户端只创建一次；空输入和退出不进入轮次处理
        mock_client.assert_called_once()
        self.assertEqual(mock_turn.call_count, 3)

        # 第三次收到的历史应保留第一轮，排除失败轮次
        self.assertEqual(
            received[2][1:],
            [
                {"role": "user", "content": "第一问"},
                {"role": "assistant", "content": "回答：第一问"},
                {"role": "user", "content": "继续问"},
            ],
        )
        self.assertIn("请输入问题", output.getvalue())
        self.assertIn("本轮未保存", output.getvalue())
        self.assertIn("对话结束", output.getvalue())
    # 检查使用最新搜索结果，失败时不退回旧候选
    def test_latest_city_candidates(self):
        history = []

        old_candidates = [
            {"name": "北京", "location": "116.41,39.92"}
        ]
        new_candidates = [
            {"name": "怀柔", "location": "116.63,40.32"}
        ]

        # 按时间顺序加入两次搜索的调用与结果
        for call_id, candidates in [
            ("city_old", old_candidates),
            ("city_new", new_candidates),
        ]:
            history.extend([
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": call_id,
                        "function": {
                            "name": "search_city_candidates"
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps({
                        "candidates": candidates
                    }),
                },
            ])

        self.assertEqual(
            agent_chat.get_latest_city_candidates(history),
            new_candidates,
        )

        # 将最新结果改为失败，不能重新启用更早的北京候选
        history[-1]["content"] = json.dumps({
            "error": "模拟城市搜索失败"
        })
        self.assertEqual(
            agent_chat.get_latest_city_candidates(history),
            [],
        )
    # 检查候选坐标允许等价写法，并拒绝列表外坐标
    def test_candidate_location_match(self):
        candidates = [
            {"name": "怀柔", "location": "116.63,40.32"}
        ]

        # 中文逗号、首尾空格和末尾零不影响数字匹配
        for location in [
            "116.63,40.32",
            " 116.630，40.320 ",
        ]:
            with self.subTest(location=location):
                agent_chat.validate_candidate_location(
                    location, candidates
                )

        # 列表外坐标必须被拒绝
        with self.assertRaisesRegex(
            ValueError, "查询坐标不属于候选列表"
        ):
            agent_chat.validate_candidate_location(
                "121.47,31.23", candidates
            )

        # 没有候选时，也不能认定匹配成功
        with self.assertRaises(ValueError):
            agent_chat.validate_candidate_location(
                "116.63,40.32", []
            )
    # 检查列表外坐标和空候选都会在执行查询前被拦截
    def test_candidate_guard_blocks_weather_request(self):
        tool_call = types.SimpleNamespace(
            id="call_outside_candidates",
            function=types.SimpleNamespace(
                name="get_current_weather",
                arguments='{"location": "121.47,31.23"}',
            ),
        )

        cases = [
            # 只有怀柔候选，不能查询其他坐标
            [{"name": "怀柔", "location": "116.63,40.32"}],

            # 搜索过但没有可用候选，也不能放行
            [],
        ]

        for candidates in cases:
            with self.subTest(candidates=candidates):
                with patch.object(
                    agent_chat,
                    "execute_weather_tool_json",
                ) as mock_execute:
                    message = agent_chat.execute_tool_call(
                        tool_call,
                        candidates=candidates,
                    )

                # 关键：校验失败后，没有进入实际工具执行入口
                mock_execute.assert_not_called()

                self.assertEqual(
                    json.loads(message["content"]),
                    {"error": "查询坐标不属于候选列表，请确认地点。"},
                )
                self.assertEqual(
                    message["tool_call_id"],
                    "call_outside_candidates",
                )
    # 检查普通确认使用历史候选，新坐标输入覆盖旧候选
    def test_agent_selects_coordinate_scope(self):
        old_candidates = [
            {"name": "怀柔", "location": "116.63,40.32"}
        ]

        cases = [
            ("选怀柔", "116.63,40.32", old_candidates),
            (
                "121.47,31.23",
                "121.47,31.23",
                [{"location": "121.47,31.23"}],
            ),
        ]

        for question, requested, expected_scope in cases:
            with self.subTest(question=question):
                tool_data = {
                    "id": "weather_scope",
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": json.dumps({
                            "location": requested
                        }),
                    },
                }
                tool_call = types.SimpleNamespace(
                    id=tool_data["id"],
                    function=types.SimpleNamespace(
                        **tool_data["function"]
                    ),
                )

                first_reply = MagicMock()
                first_reply.tool_calls = [tool_call]
                first_reply.model_dump.return_value = {
                    "role": "assistant",
                    "tool_calls": [tool_data],
                }
                final_reply = types.SimpleNamespace(
                    content="模拟回答", tool_calls=None
                )

                # 准备此前的城市搜索记录，以及当前用户输入
                messages = [
                    {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "city_scope",
                            "function": {
                                "name": "search_city_candidates"
                            },
                        }],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "city_scope",
                        "content": json.dumps({
                            "candidates": old_candidates
                        }),
                    },
                    {"role": "user", "content": question},
                ]

                with (
                    patch.object(
                        agent_chat, "request_agent_reply",
                        side_effect=[first_reply, final_reply],
                    ),
                    patch.object(
                        agent_chat, "execute_tool_call",
                        return_value={
                            "role": "tool",
                            "tool_call_id": "weather_scope",
                            "content": '{"weather": "晴"}',
                        },
                    ) as mock_execute,
                ):
                    answer = agent_chat.run_agent_turn(
                        MagicMock(), messages
                    )

                self.assertEqual(answer, "模拟回答")
                mock_execute.assert_called_once_with(
                    tool_call,
                    candidates=expected_scope,
                )
    # 检查明确编号选择、无法识别的输入及错误编号
    def test_explicit_candidate_selection(self):
        candidates = [
            {"name": "北京", "location": "116.40529,39.90499"},
            {"name": "怀柔", "location": "116.63712,40.32427"},
        ]
        original = copy.deepcopy(candidates)

        selected = agent_chat.select_candidate_by_command(
            "选第2个", candidates
        )
        self.assertEqual(selected, candidates[1])

        # 返回独立字典，不应直接修改原候选
        selected["name"] = "测试修改"
        self.assertEqual(candidates, original)

        # 不支持的自然语言不自动选择
        self.assertIsNone(
            agent_chat.select_candidate_by_command(
                "想去怀柔", candidates
            )
        )

        # 错误编号必须明确拒绝
        for question in ["选第0个", "选第9个", "选第abc个"]:
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    agent_chat.select_candidate_by_command(
                        question, candidates
                    )

        # 没有候选时也不能选择
        with self.assertRaises(ValueError):
            agent_chat.select_candidate_by_command("选第1个", [])
    # 用户选第二个时，即使模型调用第一个，也不能执行查询
    def test_agent_blocks_wrong_selected_candidate(self):
        candidates = [
            {"name": "北京", "location": "116.41,39.92"},
            {"name": "怀柔", "location": "116.63,40.32"},
        ]

        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "city_selection",
                    "function": {
                        "name": "search_city_candidates"
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "city_selection",
                "content": json.dumps({
                    "candidates": candidates
                }),
            },
            {"role": "user", "content": "选第2个"},
        ]

        # 故意模拟错误调用：请求第一个候选的坐标
        tool_data = {
            "id": "wrong_selection",
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "arguments": '{"location": "116.41,39.92"}',
            },
        }
        tool_call = types.SimpleNamespace(
            id=tool_data["id"],
            function=types.SimpleNamespace(
                **tool_data["function"]
            ),
        )

        first_reply = MagicMock()
        first_reply.tool_calls = [tool_call]
        first_reply.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [tool_data],
        }
        final_reply = types.SimpleNamespace(
            content="查询坐标与所选地点不符，未执行查询。",
            tool_calls=None,
        )

        with (
            patch.object(
                agent_chat,
                "request_agent_reply",
                side_effect=[first_reply, final_reply],
            ) as mock_reply,
            patch.object(
                agent_chat,
                "execute_weather_tool_json",
            ) as mock_execute,
        ):
            agent_chat.run_agent_turn(MagicMock(), messages)

        # 保留真实选择和校验逻辑，检查实际入口确实没被调用
        mock_execute.assert_not_called()
        self.assertEqual(mock_reply.call_count, 2)

        # 第二次模型请求收到的应是错误，而不是天气数据
        sent = mock_reply.call_args_list[1].args[1]
        self.assertEqual(sent[-1]["tool_call_id"], "wrong_selection")
        self.assertEqual(json.loads(sent[-1]["content"]), {
            "error": "查询坐标不属于候选列表，请确认地点。"
        })
    # 检查清空对话后只保留系统规则，不再携带旧候选
    def test_agent_clear_conversation(self):
        received = []
        output = io.StringIO()

        def fake_turn(client, messages):
            received.append(copy.deepcopy(messages))

            if messages[-1]["content"] == "搜索北京":
                # 模拟第一轮留下城市搜索记录
                messages.extend([
                    {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "city_before_clear",
                            "function": {
                                "name": "search_city_candidates"
                            },
                        }],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "city_before_clear",
                        "content": json.dumps({
                            "candidates": [{
                                "name": "北京",
                                "location": "116.41,39.92",
                            }]
                        }),
                    },
                ])

            messages.append({
                "role": "assistant",
                "content": "模拟回复",
            })
            return "模拟回复"

        with (
            patch.dict(sys.modules, {"openai": fake_sdk}),
            patch.object(fake_sdk, "OpenAI") as mock_client,
            patch.dict(
                os.environ,
                {"SILICONFLOW_API_KEY": "fake-key"},
            ),
            patch(
                "builtins.input",
                side_effect=[
                    "搜索北京",
                    "清空对话",
                    "清空对话",
                    "新问题",
                    "退出",
                ],
            ),
            patch.object(
                agent_chat, "run_agent_turn",
                side_effect=fake_turn,
            ) as mock_turn,
            contextlib.redirect_stdout(output),
        ):
            agent_chat.main()

        # 两次清空和退出都不进入轮次处理，也不重建客户端
        mock_client.assert_called_once()
        self.assertEqual(mock_turn.call_count, 2)

        # 清空后，新一轮只有原系统规则和新问题
        self.assertEqual(received[1], [
            received[0][0],
            {"role": "user", "content": "新问题"},
        ])
        self.assertIsNone(
            agent_chat.get_latest_city_candidates(received[1])
        )
        self.assertEqual(
            output.getvalue().count("对话已清空，可以重新开始。"),
            2,
        )
if __name__ == "__main__":
    unittest.main(verbosity=2)
