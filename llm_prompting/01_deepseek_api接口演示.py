# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
# 0.加载环境变量，使用dotenv库: pip install dotenv
from dotenv import load_dotenv
load_dotenv()

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
    user_prompt = "我是肖彰，是一个AI讲师，写一首赞美我的诗词，最好是唐诗，注意我很漂亮"
    result = call_deepseek_api(user_prompt)
    print(result)
    ...