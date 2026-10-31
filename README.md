# Python AI 旅游助手

Python AI 旅游助手学习原型，以 `main.py` 为统一命令行入口，
支持多轮聊天、旅游预算分配与校验、查看最新预算和实时天气查询。
模型使用硅基流动 API，天气使用和风天气 API。
各功能通过菜单手动启动，尚未实现模型自动调用工具。

## 当前功能

- `travel_assistant.py`：同次运行保留对话历史，支持“退出”和空输入检查。
- `budget_agent.py`：输入目的地、天数、预算；解析模型返回的 JSON；检查字段、非负整数金额及超支。
- 密钥通过环境变量读取；关闭自动重试，保留解压兼容请求头。
- 正整数输入出错后重新询问；预算中途退出可用 Ctrl+C。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| travel_assistant.py | 多轮聊天入口，修正了原文件名 trourist_assistant.py 的拼写 |
| budget_agent.py | 预算入口，移除了阻止 API 请求的临时 SystemExit |
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
E:\Anaconda\python.exe main.py
```
请在项目目录运行。统一菜单提供多轮聊天、预算生成与校验、
查看最新预算和实时天气查询。

每次聊天消息或预算生成都会请求 API，可能计费；多轮聊天会重复发送历史，输入用量随对话增长。目前不自动裁剪历史，长对话可能超过模型上下文限制。
聊天输入“退出”后不再请求；关闭程序不保存对话。网络或 API 错误会显示提示并结束本次程序，不自动恢复对话。
预算仅面向单人、不含住宿和往返目的地交通，金额单位为整数元。模型建议不等于真实票价或报价。

## 不花 API 费用的检查

```powershell
python offline_checks.py
```

该文件使用模拟客户端，不需要安装 openai，也不会发出网络请求。检查输入、JSON、校验、计算、多轮历史与退出行为；不能证明模型输出质量或真实 API 的可用性。


## 天气查询

天气模块为 `weather_service.py`，使用和风天气 API，
通过统一菜单选项 4 启动，也可以独立运行。

在 PowerShell 中设置当前窗口的环境变量：

```powershell
$env:QWEATHER_API_KEY = "填写自己的API KEY"
$env:QWEATHER_API_HOST = "填写控制台提供的主机名"
```

这里只是配置示例，不要把真实密钥写入本文件。
Host 不包含 `https://` 和接口路径。
配置仅对当前 PowerShell 窗口及其启动的程序生效。

当前支持“经度,纬度”输入，支持英文或中文逗号。
暂不支持城市 ID 和城市名称查询。
输入“退出”取消查询；空输入及不合法的坐标会在联网前被拦截。

查询成功后显示天气描述、温度及来源声明。
实际查询会访问 API，可能产生用量或费用，不自动重试。
天气功能需要单独从菜单启动，聊天模型尚不会自动调用它。

已迁移至新版 `/weather/v1/current/{latitude}/{longitude}` 接口。
用户输入顺序为“经度,纬度”，程序会转换为接口要求的“纬度/经度”。
查询结果显示天气、温度及来源声明，不再展示旧版观测时间字段。

离线检查使用模拟数据，不请求真实 API，也不覆盖真实预算：

```powershell
E:\Anaconda\python.exe offline_checks.py
```

当前共有 23 项测试，覆盖原有功能、天气解析、
模拟断网，以及取消查询和非法输入不联网等行为。
## 已知限制

- 已支持独立实时天气查询；尚无实时搜索、地图、预约或支付能力。
- 系统提示词不是事实正确性的保证；预算校验也不能核实价格是否合理。
- 模型可能追加用户未要求的追问；预算确认顺序仍依赖提示词，未由程序强制控制。
- 只清理完整的 ```json 代码块包装；其他非法输出会被拒绝。
- 模型输出达到长度上限时可能截断；不自动补全或重试。
- 无网页、数据库、持久化历史及部署入口。

## 下一阶段

当已完成新版天气接口迁移、菜单接入及一次真实查询验证，
23 项离线检查通过。后续逐步增加城市名称查询，
再探索聊天与工具的协作。

## 项目来源

基于作者提供的两份学习代码整理，使用 ChatGPT/Codex 辅助开发。保留原有主要流程，整理版本新增主函数入口、错误提示和离线检查。尚未选择开源许可证；公开前应选择适合自己的许可证。费用和模型政策不在本项目控制范围内。
