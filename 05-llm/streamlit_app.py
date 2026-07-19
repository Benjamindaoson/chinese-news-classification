"""
演示
    实现一个streamlit框架的webui, 另外一个常用的是Gradio
需要掌握的：
    1.安装streamlit: pip install streamlit
    2.启动streamlit程序: streamlit run D:\tmf_project_shanghai_AI_2\08-live_code\02-rf\streamlit_app.py
"""

# 导包
import streamlit as st
import requests
import time
# from config import Config

# 0.创建配置对象
# config = Config()

# 1.配置streamlit页面
# 设置title
st.title("投满分文本分类系统")
# 设置 输入文本 组件
text = st.text_input("请输入文本：")

# 2.向Flask后端发送请求，使用requests
# 2.1 构造网址url
url = f"http://127.0.0.1:5000/predict"
# 2.2 设置 预测按钮 的逻辑
if st.button("deepseek预测"):
    try:
        # 0.记录开始时间
        start_time = time.time()
        # 1.使用requests发送请求体，调用requests.post来获取模型预测结果
        data = {"text": text}
        result = requests.post(url, json=data)
        print(result.json())
        # 2.前端显示预测结果
        st.success(f'预测结果: {result.json()["pred_class"]} | '
                   f'推理时长: {(time.time() - start_time)*1000:.2f}ms')
    except Exception as e:
        st.error(f"出错了，请联系管理员")
        print(e)
