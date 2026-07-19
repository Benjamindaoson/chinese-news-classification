"""
演示
    通过Flask组件，构建 路由 + 函数 的应用
"""

# 导包
from flask import Flask, request, jsonify
from deepseek_predict_fun import predict_fun
# from config import Config

# 1.创建Flask应用
app = Flask(__name__)
# config = Config()

# 2.创建路由 + 视图函数
# HTTP请求方法：GET（获取资源）, POST（提交数据）, PUT（更新资源）, DELETE（删除资源）
@app.route('/predict', methods=['POST'])
def predict():
    """
    调用predict_fun函数进行预测，返回带有预测结果的字典
    :return: 带有预测结果的字典，格式{text: 文本内容,pred_class:预测类别}
    """
    # 1.获取用户请求的数据
    data = request.get_json()
    # print(f"data: {data}")
    # 2.调用predict_fun函数进行预测
    results = predict_fun(data)
    # 3.返回预测结果，使用jsonify将python对象转为json格式
    return jsonify(results)

# 3.启动应用
if __name__ == '__main__':
    # 1.只允许本机访问
    app.run(host='127.0.0.1', port=5000, debug=True)
    # # 2.允许所有设备访问
    # app.run(host='0.0.0.0', port=5000, debug=True)
    # 3.允许局域网访问
    # app.run(host='192.168.30.43', port=5000, debug=True)