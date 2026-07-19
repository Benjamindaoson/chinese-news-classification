"""
演示
    list(str)的用法，将str字符串转为一个list列表
"""

import jieba

text = "从前有座山，山上有座庙，庙里肖彰在上香。"
print(list(text))
print(jieba.lcut(text))