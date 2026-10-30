# Python AI 旅游助手

学习原型 V0.1：通过硅基流动调用模型，提供命令行多轮聊天，以及独立的旅游预算分配和校验。两个入口目前没有自动串联，不是完整的工具调用 Agent。

## 当前功能

- `travel_assistant.py`：同次运行保留对话历史，支持“退出”和空输入检查。
- `budget_agent.py`：输入目的地、天数、预算；解析模型返回的 JSON；检查字段、非负整数金额及超支。
- 密钥通过环境变量读取；关闭自动重试，保留解压兼容请求头。
- 正整数输入出错后重新询问；预算中途退出可用 Ctrl+C。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| travel_assistant.py | 命令行多轮聊天入口 |
| budget_agent.py | 预算生成与校验入口 |
| requirements.txt | 唯一直接依赖 openai，版本沿用作者已成功运行的版本 |
| .gitignore | 排除密钥、本地 IDE 配置、缓存和虚拟环境 |
| offline_checks.py | 模拟 API 的离线检查，无密钥、无网络请求 |
| docs/TEST_RECORD.md | 本次整理的检查记录与限制 |

## 安装

建议 Python 3.12。作者原环境为 Windows / Anaconda Python 3.12.4。
在项目目录的终端执行（python 必须指向你实际使用的解释器）：

```powershell
python -m pip install -r requirements.txt
```

如果使用原来的解释器，也可执行：

```powershell
E:\Anaconda\python.exe -m pip install -r requirements.txt
```

## 配置密钥

PyCharm：Run → Edit Configurations → 选择实际运行的文件 → Environment variables，添加 `SILICONFLOW_API_KEY`。两个运行配置需要分别设置。

或在 PowerShell 为当前终端临时设置：

```powershell
$env:SILICONFLOW_API_KEY = "在自己的电脑上填写真实密钥"
```

不要将带真实密钥的命令、截图或配置提交到 GitHub。不提供 .env 自动加载功能。
默认模型 ID 为 `Qwen/Qwen3-32B`，可通过可选环境变量 `SILICONFLOW_MODEL` 修改。是否可用及计费以平台控制台为准。

## 运行

```powershell
python travel_assistant.py
python budget_agent.py
```

每次聊天消息或预算生成都会请求 API，可能计费；多轮聊天会重复发送历史，输入用量随对话增长。目前不自动裁剪历史，长对话可能超过模型上下文限制。
聊天输入“退出”后不再请求；关闭程序不保存对话。网络或 API 错误会显示提示并结束本次程序，不自动恢复对话。
预算仅面向单人、不含住宿和往返目的地交通，金额单位为整数元。模型建议不等于真实票价或报价。

## 不花 API 费用的检查

```powershell
python offline_checks.py
```

该文件使用模拟客户端，不需要安装 openai，也不会发出网络请求。检查输入、JSON、校验、计算、多轮历史与退出行为；不能证明模型输出质量或真实 API 的可用性。

## 已知限制

- 无实时搜索、地图、天气、预约或支付能力；具体场馆、政策和路线可能不准确。
- 系统提示词不是事实正确性的保证；预算校验也不能核实价格是否合理。
- 模型可能追加用户未要求的追问；预算确认顺序仍依赖提示词，未由程序强制控制。
- 只清理完整的 ```json 代码块包装；其他非法输出会被拒绝。
- 模型输出达到长度上限时可能截断；不自动补全或重试。
- 无网页、数据库、持久化历史及部署入口。

## 下一阶段

已完成本机离线检查和两个入口的真实调用验证。下一阶段整合对话和预算模块，添加界面及真实工具调用。

## 项目来源

基于作者提供的两份学习代码整理，使用 ChatGPT/Codex 辅助开发。保留原有主要流程，整理版本新增主函数入口、错误提示和离线检查。本仓库当前未选择开源许可证；公开可见不等于授予开源许可。费用和模型政策不在本项目控制范围内。
