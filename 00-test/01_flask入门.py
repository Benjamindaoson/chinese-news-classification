"""
演示
    flask框架入门操作，创建一个简单的API服务
需要掌握:
    安装flask: pip install flask
    app = Flask(__name__)   # 创建Flask应用
    @app.route('/', methods=['GET', 'POST'])    # 创建路由
    def predict(): 视图函数
"""
# Flask类可以创建web应用，send_file发送文件响应
from flask import Flask, send_file, render_template
# 1.创建Flask应用
app = Flask(__name__)

# 2.创建路由
@app.route('/', methods=['GET'])
# HTTP请求方法: GET(获取资源), POST(提交数据), PUT(更新资源), DELETE(删除资源)
# URL路径: http://host:port/route_name
def predict():
    return "hello world"

# 3.启动服务
if __name__ == '__main__':
    # 1.只允许本机访问
    app.run(host='127.0.0.1', port=5000, debug=True)
    # # 2.允许所有设备访问
    # app.run(host='0.0.0.0', port=5000, debug=True)
    # 3.允许局域网访问
    # app.run(host='192.168.0.100', port=5000, debug=True)
