"""
演示
    使用提示词工程完成文本分类任务
注意：
    系统提示词是由AI大模型生成的。我们只需要写提示词的提示词，需要写的尽可能详细。
"""

# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
# 0.加载环境变量，使用dotenv库: pip install dotenv
from dotenv import load_dotenv
load_dotenv()

# 系统提示词
SYSTEM_PROMPT = """
你是一个文本分类助手。根据输入的文本，从以下类别中选择一个最合适的类别：finance, realty, stocks, education, science, society, politics, sports, game, entertainment。只输出类别名称，不要输出索引或任何其他额外内容。仅输出一个类别名称。

"""

# 定义函数，调用DeepSeek API获取回复
# system_prompt: 告诉系统的角色，让它做什么，类似cosplay
# user_prompt: 用户提问，也就是用户的实际的问题，输入给大模型，让大模型返回结果
def call_deepseek_api(user_prompt, system_prompt="You are a helpful assistant"):
    # 1.创建OpenAI客户端，使用环境变量中的API Key
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")

    # 2.调用聊天接口，发送消息并获取回复
    response = client.chat.completions.create(
        model="deepseek-v4-flash",  # 模型名称，支持deepseek-v4-flash，deepseek-v4-pro
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],  # 消息列表
        stream=False,   # 流式输出，默认为False
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )

    return response.choices[0].message.content

# 测试
if __name__ == "__main__":
    user_prompt = "涉股理财品PK基金正规军"
    result = call_deepseek_api(user_prompt, system_prompt=SYSTEM_PROMPT)
    print(result)
    ...