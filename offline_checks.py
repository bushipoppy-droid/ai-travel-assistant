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
        # 收集菜单输出，避免测试时反复打印菜单
        output = io.StringIO()

        # 模拟依次选择无效选项、四个功能，最后退出
        # 各功能都被临时替换，不会真正聊天、生成预算或查询天气
        with patch("builtins.input", side_effect=["9", "1", "2", "3", "4", "0"]), \
                patch.object(chat, "main") as chat_call, \
                patch.object(budget, "main") as budget_call, \
                patch.object(menu, "show_latest_budget") as view_call, \
                patch.object(menu.weather_service, "main") as weather_call, \
                contextlib.redirect_stdout(output):
            menu.main()

        # 检查每个功能是否恰好调用一次
        chat_call.assert_called_once()
        budget_call.assert_called_once()
        view_call.assert_called_once()
        weather_call.assert_called_once()

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
        self.assertEqual(len(messages), 1)
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
if __name__ == "__main__":
    unittest.main(verbosity=2)
