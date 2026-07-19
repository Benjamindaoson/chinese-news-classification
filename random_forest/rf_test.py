"""
演示
    使用验证集，评估 随机森林模型
工作流:
    1.加载数据集
    2.加载模型和向量化器
    3.文本数值化:使用TF-IDF提取文本特征
    4.模型评估
"""

# 导包
import pandas as pd
import pickle   # 用于保存模型,sklearn模型可以使用pickle进行序列化和反序列化
# 模型评估指标：准确率、精确率、召回率、F1-score、分类评估报告、混淆矩阵
# F1-score = 2P*R/(P+R), P是精确率，R是召回率
# confusion_matrix: 混淆矩阵，矩阵的行表示真实类别，列表示预测类别
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from config import Config

# 0.pandas的基础设置
pd.set_option('display.expand_frame_repr', False)  # 避免宽表格换行
pd.set_option('display.max_columns', None)  # 确保所有列可见
# 创建配置对象
config = Config()

# 1.加载数据集, 验证集
dev_data = pd.read_csv(config.process_dev_path)
print(dev_data.shape)
print(dev_data.head())
# 2.加载模型和向量化器
with open(config.rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)
with open(config.tfidf_model_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
# 3.文本数值化,TF-IDF向量化器 TfidfVectorizer
# 1.提取分词后的结果words
words = dev_data['words']
# 2.提取标签列label
labels = dev_data['label']
# 3.使用TF-IDF向量化器对象进行向量化, transform
words_features = tfidf_vectorizer.transform(words)
print(words_features.shape)

# 4.模型评估
y_pred = rf_model.predict(words_features)
print(f"预测结果y_pred.shape：{y_pred.shape}")
# 计算acc = 预测正确的数量 / 总样本量
acc = accuracy_score(labels, y_pred)
# 解释 macro / micro / weighted
# macro(更加考虑小类别): 宏平均，计算各个类别指标，然后直接算平均值，每个类别权重一样
# weighted(用的最多): 加权平均，计算各个类别指标，然后根据每个类别样本数量进行加权平均，每个类别权重不一样
# micro(用的很少): 微平均，汇总每个类别的 TP/FP/FN,然后统一计算指标，大类权重更大
# 计算precision = TP / (TP+FP)
precision = precision_score(labels, y_pred, average='macro')
print(f"precision:{precision}")
# 计算recall = TP / (TP+FN)
recall = recall_score(labels, y_pred, average='macro')
print(f"recall:{recall}")
# 计算F1-score = 2*p*r/(p+r)
f1 = f1_score(labels, y_pred, average='macro')
print(f"f1:{f1}")
# 分类评估报告
print(classification_report(labels, y_pred))
# 混淆矩阵
print(confusion_matrix(labels, y_pred))
