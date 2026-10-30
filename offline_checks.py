"""离线检查：模拟API，不使用真实密钥，不访问网络。"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import patch


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
        output = io.StringIO()
        with patch("builtins.input", side_effect=["9", "1", "2", "3", "0"]), \
             patch.object(chat, "main") as chat_call, \
             patch.object(budget, "main") as budget_call, \
             patch.object(menu, "show_latest_budget") as view_call, \
             contextlib.redirect_stdout(output):
            menu.main()
        chat_call.assert_called_once()
        budget_call.assert_called_once()
        view_call.assert_called_once()
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
