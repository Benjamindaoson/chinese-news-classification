"""
    数据预处理，进行jieba分词，保存为CSV
"""

# 导包
import pandas as pd # 处理表格数据
import jieba # 中文分词 pip install jieba
from config import Config # 配置文件
import matplotlib.pyplot as plt # 数据可视化
import seaborn as sns
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False
# 1.创建配置对象
config = Config()
# 2.定义函数，处理数据，进行jieba分词，保存为CSV
def process_data(data_path, save_path):
    """
    数据处理函数，对原始数据集进行jieba分词并保存为CSV
    :param data_path: 原始数据集路径，txt格式，数据格式为: text \t label
    :param save_path: 处理后数据集的保存路径，CSV格式，包含列: text, label
    :return: 无
    """
    # 1.读取原始文件
    data = pd.read_csv(data_path, sep='\t', names=['text', 'label'])
    # 2.进行jieba分词
    def cut_words(text):
        return " ".join(jieba.lcut(text)[:20])
    data['words'] = data['text'].apply(cut_words)
    # 3.获取序列长度
    def get_words(text):
        return len(jieba.lcut(text))
    data['seq_len'] = data['text'].apply(get_words)
    # print(data.head())
    # print(data['seq_len'].describe())
    # 4.可视化序列长度分布
    data['seq_len'].hist()
    plt.title("序列长度分布")
    plt.show()
    # 5.保存处理后的数据为CSV
    data.to_csv(save_path, index=False)
    print(f"处理后数据保存在{save_path}中")
    ...

# 3.处理数据
process_data(config.train_path, config.process_train_path)
process_data(config.test_path, config.process_test_path)
process_data(config.dev_path, config.process_dev_path)