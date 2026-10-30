import os
import json
from pathlib import Path
from openai import OpenAI, APIError


# 1. 检查必需字段是否齐全
def validate_keys(costs):
    required_keys = ["ticket", "transport", "food", "other"]

    for key in required_keys:
        if key not in costs:
            return False

    return True


# 2. 检查金额是否都是非负整数
def validate_amounts(costs):
    for key, value in costs.items():
        if type(value) is not int or value < 0:
            print("不合法的费用项目：", key)
            print("收到的金额：", value)
            return False

    return True


# 3. 由Python计算总费用和剩余预算
def check_budget(ticket, transport, food, other, budget):
    total = ticket + transport + food + other
    return total, budget - total


# 4. 主流程：接收需求、调用模型、检查并保存预算
def main():
    # 读取环境变量中的密钥
    api_key = os.getenv("SILICONFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "未读取到密钥，请配置 SILICONFLOW_API_KEY 环境变量。"
        )

    # 创建客户端，不自动重试
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        timeout=60.0,
        max_retries=0,
        default_headers={"Accept-Encoding": "identity"},
    )

    # 5. 接收目的地
    destination = input("请输入旅游目的地：").strip()

    if not destination:
        raise ValueError("目的地不能为空")

    # 6. 循环接收天数，直到输入正整数
    while True:
        try:
            days = int(input("请输入游玩天数："))
        except ValueError:
            print("请输入整数，例如3，不要输入“三天”。")
            continue

        if days <= 0:
            print("游玩天数必须大于0。")
            continue

        break

    # 7. 循环接收预算，直到输入正整数
    while True:
        try:
            budget = int(input("请输入旅游预算上限（元）："))
        except ValueError:
            print("请输入整数金额，例如1000。")
            continue

        if budget <= 0:
            print("预算必须大于0。")
            continue

        break

    # 8. 设置模型任务和JSON输出要求
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名旅游预算分配助手。"
                "根据用户提供的目的地、天数和预算，"
                "为单人旅行分配费用，不含住宿和往返目的地的交通。"
                "金额仅为规划建议，不是查询到的实际价格。"
                "只返回JSON对象，不要解释，不要使用Markdown代码块。"
                "只包含ticket、transport、food、other四个键，"
                "分别表示门票、市内交通、餐饮和其他费用。"
                "金额必须是非负整数，优先在用户预算内分配。"
                "不必花完全部预算。"
                "如果预算可能不足以支持旅行，"
                "不要为了凑预算而给出明显不合理的低金额；"
                "可以给出超预算的建议分配，由程序检查并提示超支。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"目的地：{destination}；"
                f"游玩天数：{days}天；"
                f"单人预算上限：{budget}元。"
                "请分配四项费用。"
            ),
        },
    ]

    # 9. 请求模型，本次只调用一次
    response = client.chat.completions.create(
        model=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-32B"),
        messages=messages,
        max_tokens=256,
        extra_body={"enable_thinking": False},
    )

    # 10. 取出模型回复，检查是否被截断
    text = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason

    print("\n模型原始输出：", text)
    print("结束原因：", finish_reason)

    if finish_reason == "length":
        print("模型回复可能已截断，本次不解析、不保存。")
        return

    # 11. 清理完整包裹在外面的JSON代码块
    if isinstance(text, str):
        text = text.strip()

        if text.startswith("```json") and text.endswith("```"):
            text = text[len("```json"):-len("```")].strip()

    # 12. 将JSON文本解析为Python数据
    try:
        costs = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        print("解析错误：模型没有返回可解析的JSON文本。")
        return

    # 13. 按顺序检查类型、字段和金额
    # 检查失败就结束本次功能，不覆盖已有预算文件
    if not isinstance(costs, dict):
        print("格式错误：解析结果必须是字典。")
        return

    if not validate_keys(costs):
        print("字段错误：缺少必需的费用项目。")
        return

    if len(costs) != 4:
        print("字段错误：只能包含四个指定费用项目。")
        return

    if not validate_amounts(costs):
        print("金额错误：费用必须是非负整数。")
        return

    # 14. 全部数据检查通过后，由Python计算预算
    total_cost, remaining = check_budget(
        costs["ticket"],
        costs["transport"],
        costs["food"],
        costs["other"],
        budget,
    )

    print("门票预算：", costs["ticket"], "元")
    print("市内交通：", costs["transport"], "元")
    print("餐饮预算：", costs["food"], "元")
    print("其他费用：", costs["other"], "元")
    print("总费用：", total_cost, "元")

    if remaining >= 0:
        print("剩余预算：", remaining, "元")
    else:
        print("超出预算：", -remaining, "元")

    # 15. 整理本次预算，标明范围和是否超支
    result = {
        "destination": destination,
        "days": days,
        "budget": budget,
        "currency": "CNY",
        "travelers": 1,
        "includes_accommodation": False,
        "includes_round_trip_transport": False,
        "costs": costs,
        "total_cost": total_cost,
        "remaining": remaining,
        "over_budget": remaining < 0,
        "note": "费用为模型建议分配，未经实时价格核实。",
    }

    # 16. 固定保存到budget_agent.py所在目录
    save_path = Path(__file__).resolve().parent / "budget_result.json"
    temp_path = save_path.with_suffix(".json.tmp")

    # 先写临时文件，完成后替换正式文件
    # 避免写入中途失败时直接破坏旧预算文件
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        os.replace(temp_path, save_path)

    except OSError:
        print("预算保存失败，请检查目录权限和磁盘空间。")

        # 尝试清理未完成的临时文件
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    else:
        print("最新预算已保存到：", save_path)


# 17. 直接运行本文件时才启动；导入时不调用API
if __name__ == "__main__":
    try:
        main()

    except APIError:
        print(
            "API请求失败：请检查网络、余额、密钥权限和模型ID。"
            "本次未自动重试。"
        )

    except (RuntimeError, ValueError) as error:
        print(error)

    except (KeyboardInterrupt, EOFError):
        print("\n已结束程序。")