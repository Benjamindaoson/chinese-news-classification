"""
测试api_flask_server中的Flask后端API服务是否正常
"""

# 导包
import requests
from config import Config

# 1.创建配置对象
config = Config()
# 2.定义接口地址URL
teacher_url = f"http://{config.api_host}:{config.api_port}/predict1"
student_url = f"http://{config.api_host}:{config.api_port}/predict2"
# 3.测试调用Flask后端API服务
try:
    # 1.测试调用接口
    text = input("请输入文本内容：")
    # 2.使用requests来发送请求体
    data = {"text": text}
    teacher_result = requests.post(teacher_url, json=data)
    student_result = requests.post(student_url, json=data)
    # 3.打印结果
    print(f"教师模型预测结果：{teacher_result.json()}")
    print(f"学生模型预测结果：{student_result.json()}")

except Exception as e:
    print(f"出问题了,请联系管理员, {e}!")
