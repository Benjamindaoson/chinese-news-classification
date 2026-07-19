"""
测试api_flask_server中的Flask后端API服务是否正常
"""

# 导包
import requests
from config import Config

# 1.创建配置对象
config = Config()
# 2.定义接口地址URL
url = f"http://{config.api_host}:{config.api_port}/predict"
# 3.测试调用Flask后端API服务
try:
    # 1.测试调用接口
    text = input("请输入文本内容：")
    # 2.使用requests来发送请求体
    data = {"text": text}
    r = requests.post(url, json=data)
    # 3.打印结果
    print(f"预测结果：{r.json()}")

except Exception as e:
    print(f"出问题了,请联系管理员, {e}!")