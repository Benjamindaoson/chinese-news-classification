"""
演示
    使用Path(__file__)获取当前文件路径
    比如D:\tmf_project_shanghai_AI_2\chinese-news-classification\data\10_扩展_演示Path.py

"""

from pathlib import Path  # 路径操作模块
import os
print(os.path.join(Path(__file__).parent.parent,"data"))
# root_path = os.path.join(Path(__file__).parent.parent, "data")

print(os.path.dirname(Path(__file__)))
