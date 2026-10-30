import os
from openai import OpenAI, APIError


# 主流程：在本次运行中保留历史，循环接收问题
def main():
    # 1. 读取密钥，不在代码里直接填写密钥
    api_key = os.getenv("SILICONFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "未读取到密钥，请检查当前运行配置中的 SILICONFLOW_API_KEY"
        )

    # 2. 创建客户端，连接硅基流动
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        timeout=60.0,
        max_retries=0,
        default_headers={"Accept-Encoding": "identity"},
    )

    # 3. 在循环外创建消息列表，保留本次运行中的对话历史
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名旅游规划助手，回复简短。"
                "每轮最多问两个问题，不重复询问已明确的信息。"
                "优先执行预算确认规则："
                "如果用户提供预算并说酒店已订，"
                "但没有明确预算是否包含住宿，"
                "本轮只询问这笔预算是否包含已经支付的酒店费用，"
                "不询问交通方式或兴趣细分，不生成行程。"
                "用户已明确预算范围时，不重复确认，继续处理用户需求。"
                "你没有联网工具，不得声称已查询实时信息。"
                "没有查询依据时，不提供具体票价、开放时间、"
                "预约政策、交通线路编号或换乘步骤。"
            ),
        },
    ]

    print("旅游助手已启动，输入“退出”结束对话。")

    # 4. 循环接收问题，直到用户输入“退出”
    while True:
        question = input("\n你：").strip()

        # 先判断是否退出，避免把“退出”发送给模型
        if question == "退出":
            break

        # 如果没有输入内容，直接进入下一轮，不调用模型
        if not question:
            print("请输入问题，或输入“退出”。")
            continue

        # 5. 将本轮问题添加为用户消息
        messages.append({
            "role": "user",
            "content": question,
        })

        # 6. 发送整个消息列表，让模型看到之前的对话
        response = client.chat.completions.create(
            model=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-32B"),
            messages=messages,
            max_tokens=512,
            extra_body={"enable_thinking": False},
        )

        # 7. 取出并打印模型回复
        answer = response.choices[0].message.content
        if not answer:
            print("本轮未收到文字回复，程序结束。")
            return
        print("助手：", answer)

        # 8. 将回复添加为助手消息，供下一轮使用
        messages.append({
            "role": "assistant",
            "content": answer,
        })

        # 9. 查看当前消息数量和本轮结束原因
        print("当前消息数量：", len(messages))
        print("结束原因：", response.choices[0].finish_reason)
        if response.choices[0].finish_reason == "length":
            print("提示：回复可能已截断，请缩小问题范围。")

    # 10. 离开循环后，结束程序
    print("对话结束")


# 直接运行时才启动；导入函数不会询问输入或调用API
if __name__ == "__main__":
    try:
        main()
    except APIError:
        # 不自动重试，也不打印完整请求异常
        print("API请求失败：请检查网络、余额、密钥权限和模型ID。本次未自动重试。")
    except (RuntimeError, ValueError) as error:
        print(error)
    except (KeyboardInterrupt, EOFError):
        print("\n已结束程序。")
