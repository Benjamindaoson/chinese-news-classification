"""
EDA(Exploratory Data Analysis):
    探索性数据分析,帮助发现数据规律或异常，一般集合图标等可视化形式
    就是分析数据，来查看标签数量分布、句子长度分布
需要掌握的：
    series.hist():绘制分布直方图
"""
# 导包
import pandas as pd # 处理表格数据
from collections import Counter # 统计标签的频次
import matplotlib.pyplot as plt # 可视化工具
import seaborn as sns # 可视化工具
from config import Config # 导入配置类
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False            # 解决负号显示问题
# 1.初始化配置类
config = Config()
# 2.读取数据
train_data = pd.read_csv(config.train_path, sep="\t", names=["text", "label"])
dev_data = pd.read_csv(config.dev_path, sep="\t", names=["text", "label"])
test_data = pd.read_csv(config.test_path, sep="\t", names=["text", "label"])
print(train_data.shape) # (179000, 2)
print(dev_data.shape)   # (9993, 2)
print(test_data.shape)  # (9990, 2)
print(train_data.head())
print("-"*40)
# 工程上要求对训练集进行统计，防止数据泄露。这里为了演示，同时统计了训练集、验证集、测试集
# 3.统计标签分布
label_counts = Counter(train_data["label"])
print(f"标签分布：{label_counts}")
# 4.统计句子长度
# 使用apply方式获取句子长度
# train_data["text_length"] = train_data['text'].apply(len)
# 使用pandas向量化操作str.len来获取句子长度
train_data["text_length"] = train_data['text'].str.len()
print(train_data.head())
print(train_data["text_length"].describe())
# 5.绘制标签分布图
# sns.countplot(x="label",data=train_data, hue="label")
train_data['label'].hist()
plt.title("标签分布")
plt.show()
# 6.绘制句子长度分布图
# sns.histplot(train_data["text_length"],bins=20,kde=True)
train_data['text_length'].hist()
plt.title("句子长度分布")
plt.show()