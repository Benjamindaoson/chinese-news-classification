"""
演示
    flask框架入门操作，创建一个简单的API服务
需要掌握:
    安装flask: pip install flask
    app = Flask(__name__)
    @app.route('/', methods=['GET', 'POST'])
"""
# Flask类可以创建web应用，send_file发送文件响应
from flask import Flask, send_file, render_template_string

# 1.创建Flask应用
app = Flask(__name__)

# 定义首页模板
HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>投满分项目</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f2f5;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            text-align: center;
            background-color: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            width: 90%;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        .btn {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
            text-decoration: none;
            display: inline-block;
        }
        .btn:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>欢迎来到投满分项目</h1>
        <a href="/show_image" class="btn">开始</a>
    </div>
</body>
</html>
"""

# 定义图片展示页模板
IMAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>图片展示</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f2f5;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .image-container {
            margin-bottom: 20px;
            max-width: 90%;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .btn {
            background-color: #6c757d;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
            text-decoration: none;
        }
        .btn:hover {
            background-color: #545b62;
        }
    </style>
</head>
<body>
    <div class="image-container">
        <img src="/get_image" alt="展示图片">
    </div>
    <a href="/" class="btn">返回</a>
</body>
</html>
"""

# 2.创建路由

# 首页路由
@app.route('/', methods=['GET'])
def home():
    return render_template_string(HOME_TEMPLATE)

# 展示图片页面路由
@app.route('/show_image', methods=['GET'])
def show_image_page():
    return render_template_string(IMAGE_TEMPLATE)

# 获取图片数据的路由
@app.route('/get_image', methods=['GET'])
def get_image():
    return send_file("img/img.jpg")

# 3.启动服务
if __name__ == '__main__':
    # 1.只允许本机访问
    app.run(host='127.0.0.1', port=5000, debug=True)
    # # 2.允许所有设备访问
    # app.run(host='0.0.0.0', port=5000, debug=True)
    # 3.允许局域网访问
    # app.run(host='192.168.0.100', port=5000, debug=True)
