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


class FakeClient:
    # 记录每次请求的消息快照，用于验证历史是否真的传入
    def __init__(self, text):
        self.text = text
        self.requests = []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content=self.text), finish_reason="stop"
        )
        return types.SimpleNamespace(choices=[choice])


def run_budget(text, inputs=None):
    client = FakeClient(text)
    output = io.StringIO()
    with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "offline-placeholder"}), \
         patch.object(budget, "OpenAI", return_value=client), \
         patch("builtins.input", side_effect=inputs or ["深圳", "3", "1500"]), \
         contextlib.redirect_stdout(output):
        budget.main()
    return output.getvalue(), client


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
