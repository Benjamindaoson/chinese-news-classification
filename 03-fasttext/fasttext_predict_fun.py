"""
演示
    模型预测,这里采用ft_model_char_1.bin(按字分词+手动调参)
涉及到的API:
    fasttext.load_model
    model.predict
"""

# 导包
import fasttext
import time
from config import Config

# 取消警告显示
import warnings
warnings.filterwarnings("ignore")

# ===================== ✅ 唯一有效方案：直接拦截numpy错误行为 =====================
import numpy as np
# 临时覆盖np.array，强制把copy=False改成copy=True，只修复这一处
old_array = np.array
def patched_array(obj, copy=True, *args, **kwargs):
    if copy is False:  # 拦截fasttext的坏调用
        copy = True
    return old_array(obj, copy=copy, *args, **kwargs)
np.array = patched_array
# =====================
# 1.创建配置对象
config = Config()

# 2.加载模型
model = fasttext.load_model(config.ft_model_path.replace(".bin", "_char_1.bin"))
# 3.定义函数，实现模型预测
def predict_fun(data):
    """
    这是模型预测，根据传入的字典，调用模型进行预测得到最终类别，并返回字典
    :param data: 字典，格式{text: 文本内容}
    :return: 字典，包含预测结果，格式{text: 文本内容, pred_class:预测类别}
    """
    # 1.分词，按字分词
    text = data['text']
    cut_words = " ".join(list(text))
    # 2.模型预测，model.predict
    y_pred = model.predict(cut_words)
    # y_pred: (标签，概率)
    # 3.获取预测类别名称
    data['pred_class'] = y_pred[0][0].replace("__label__", "")
    # 4.返回结果
    return data

# 测试
if __name__ == '__main__':
    data = {
        "text": "恒指下午跌幅收窄 可留意牛64357熊65110"
    }
    results = predict_fun(data)
    print(results)
    # 计算推理延迟：单条样本的推理时长
    start_time = time.time()
    for i in range(200):
        result = predict_fun(data)
    total_time = time.time() - start_time
    print(f"推理延迟为：{total_time / 200 * 1000:.3f}ms")
    ...





