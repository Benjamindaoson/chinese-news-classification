"""
演示
    随机森林模型的 模型预测
"""

# 导包
import jieba    # 中文分词工具
from config import Config
import pickle
import time
# 取消警告显示
import warnings
warnings.filterwarnings("ignore")

# 1.加载配置文件
config = Config()

# 2.加载模型和向量化器
with open(config.rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)
with open(config.tfidf_model_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# 3.定义函数，实现模型预测
def predict_fun(data):
    """
    这是模型预测，根据传入的字典，调用模型进行预测得到最终类别，并返回字典
    :param data: 字典，格式{text: 文本内容}
    :return: 字典，包含预测结果，格式{text: 文本内容,pred_class:预测类别}
    """
    # 1.对输入文本进行分词
    words = " ".join(jieba.lcut(data['text'])[:20])
    # 2.进行TF-IDF向量化
    tfidf_data = tfidf_vectorizer.transform([words])
    # 3.模型预测，得到预测类别索引
    y_pred = rf_model.predict(tfidf_data)[0]    # (1,)
    # 4.获取预测类别名称
    # 构造类别索引-名称的映射字典
    with open(config.class_path, 'r') as f:
        # 1.使用for循环来遍历行，去掉空白符，内存占用小
        class_list = [line.strip() for line in f if line.strip()]
    id2class = {i:name for i,name in enumerate(class_list)}
    # print(id2class)
    y_pred_class = id2class[y_pred]
    data['pred_class'] = y_pred_class
    # 5.返回结果
    return data

# 测试
if __name__ == '__main__':
    data = {
        'text': '从前有一段真挚的爱情放在肖章的面前，可惜他当时正在考四级，如果再给他一次机会，他一定还会考四级'
    }
    result = predict_fun(data)
    print(result)
    # 计算推理延迟：单条样本的推理时长
    start_time = time.time()
    for i in range(200):
        result = predict_fun(data)
    total_time = time.time() - start_time
    print(f"推理延迟为：{total_time / 200 * 1000:.2f}ms")
    ...