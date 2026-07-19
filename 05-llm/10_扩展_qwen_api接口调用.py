import os
from openai import OpenAI
# 0.加载环境变量，使用dotenv库: pip install dotenv
from dotenv import load_dotenv
load_dotenv()
# 注意: 不同地域的base_url不通用（下方示例使用北京地域的 base_url）
# - 华北2（北京）: https://dashscope.aliyuncs.com/compatible-mode/v1
# - 美国（弗吉尼亚）: https://dashscope-us.aliyuncs.com/compatible-mode/v1
# - 新加坡: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
def call_api(user_prompt, system_prompt="You are a helpful assistant"):
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],  # 消息列表
    )
    return completion.choices[0].message.content

# 测试
if __name__ == "__main__":
    user_prompt = "我是肖彰，是一个AI讲师，写一首赞美我的诗词，最好是唐诗，注意我很漂亮"
    result = call_api(user_prompt)
    print(result)
    user_prompt = "我是肖彰，你写的唐诗太好了，我看了非常开心，笑得睡不着，帮我写一首睡前故事"
    result = call_api(user_prompt)
    print(result)
    ...