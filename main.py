import json
from pathlib import Path
from openai import APIError

import travel_assistant
import budget_agent
# 导入天气模块；导入时不会自动查询
import weather_service

# 1. 读取并展示本地保存的最新预算，不调用API
def show_latest_budget():
    # 与budget_agent.py使用同一个保存目录
    save_path = (
        Path(budget_agent.__file__).resolve().parent
        / "budget_result.json"
    )

    # 首次使用时，预算文件可能还不存在
    if not save_path.exists():
        print("还没有保存的预算，请先使用功能2生成预算。")
        return

    # 读取JSON文件，转换为Python数据
    try:
        with open(save_path, "r", encoding="utf-8") as file:
            result = json.load(file)

    except (OSError, UnicodeError, json.JSONDecodeError):
        print("预算文件读取失败，请检查文件是否损坏或无法访问。")
        return

    # 展示前确认数据结构，防止文件被手动修改后报错
    if not isinstance(result, dict):
        print("预算文件格式错误：内容必须是JSON对象。")
        return

    required_keys = [
        "destination",
        "days",
        "budget",
        "costs",
    ]

    for key in required_keys:
        if key not in result:
            print("预算文件缺少字段：", key)
            return

    costs = result["costs"]

    if not isinstance(costs, dict):
        print("预算文件格式错误：costs必须是字典。")
        return

    if not budget_agent.validate_keys(costs):
        print("预算文件缺少必需的费用项目。")
        return

    if len(costs) != 4 or not budget_agent.validate_amounts(costs):
        print("预算文件中的费用项目或金额不合法。")
        return

    if type(result["budget"]) is not int or result["budget"] <= 0:
        print("预算文件中的预算上限不合法。")
        return

    # 重新计算，避免只相信文件里保存的总额
    total, remaining = budget_agent.check_budget(
        costs["ticket"],
        costs["transport"],
        costs["food"],
        costs["other"],
        result["budget"],
    )

    print("\n===== 最新保存的预算 =====")
    print("目的地：", result["destination"])
    print("游玩天数：", result["days"])
    print("预算上限：", result["budget"], "元")
    print("门票预算：", costs["ticket"], "元")
    print("市内交通：", costs["transport"], "元")
    print("餐饮预算：", costs["food"], "元")
    print("其他费用：", costs["other"], "元")
    print("总费用：", total, "元")

    if remaining >= 0:
        print("剩余预算：", remaining, "元")
    else:
        print("超出预算：", -remaining, "元")

    print("范围：单人，不含住宿和往返交通。")
    print("提示：费用为模型建议分配，未经实时价格核实。")


# 2. 统一菜单，功能结束后返回这里
def main():
    while True:
        print("\n===== AI 旅游助手 =====")
        print("1. 多轮聊天")
        print("2. 旅游预算分配与校验")
        print("3. 查看最新预算（不调用API）")
        print("4. 实时天气查询（调用天气API）")
        print("0. 退出程序")

        choice = input("请选择功能（1/2/3/4/0）：").strip()

        if choice == "0":
            print("旅游助手已关闭。")
            break

        try:
            if choice == "1":
                travel_assistant.main()

            elif choice == "2":
                budget_agent.main()

            elif choice == "3":
                show_latest_budget()
            elif choice == "4":
                # 调用天气主流程，完成后继续显示菜单
                weather_service.main()

            else:
                print("选项无效，请输入1、2、3、4或0。")

        except APIError:
            print("API请求失败，请检查网络、余额、密钥权限和模型ID。")

        except (RuntimeError, ValueError) as error:
            print("运行提示：", error)


# 3. 直接运行本文件时启动菜单
if __name__ == "__main__":
    try:
        main()

    except (KeyboardInterrupt, EOFError):
        print("\n旅游助手已关闭。")