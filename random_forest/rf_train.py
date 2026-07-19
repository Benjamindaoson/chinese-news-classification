"""
演示
    训练 随机森林 和 TF-IDF向量化器
工作流:
    1.加载数据集
    2.文本数值化:使用TF-IDF提取文本特征
    3.训练随机森林模型
    4.模型评估
    5.保存模型
需要掌握的：
    f1_score(y_test, y_pred, average='macro')
"""

# 导包
import pandas as pd
import pickle   # 用于保存模型,sklearn模型可以使用pickle进行序列化和反序列化
from sklearn.feature_extraction.text import TfidfVectorizer # 用于文本特征提取，TF-IDF向量化器
from sklearn.model_selection import train_test_split    # 用于划分训练集和验证集
# 模型评估指标：准确率、精确率、召回率、F1-score、分类评估报告、混淆矩阵
# F1-score = 2P*R/(P+R), P是精确率，R是召回率
# confusion_matrix: 混淆矩阵，矩阵的行表示真实类别，列表示预测类别
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from config import Config
# 集成学习模型：随机森林、AdaBoost、GBDT
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from tqdm import tqdm
# 0.pandas的基础设置
pd.set_option('display.expand_frame_repr', False)  # 避免宽表格换行
pd.set_option('display.max_columns', None)  # 确保所有列可见
# 创建配置对象
config = Config()

# 1.加载数据集, 训练集的前20000行
train_data = pd.read_csv(config.process_train_path)[:20000]
print(train_data.shape)
print(train_data.head())
# 2.文本数值化,训练TF-IDF向量化器 TfidfVectorizer
# 1.提取分词后的结果words
words = train_data['words']
# 2.提取标签列label
labels = train_data['label']
# 3.读取停用词文件
with open(config.stopwords_path, 'r', encoding='utf-8') as f:
    # 1.使用for循环遍历行,去掉空白符
    stopwords = [line.strip() for line in f if line.strip()]
    # 2.直接使用f.read().split()来一次性获取所有行,内存占用大
    # stopwords = f.read().split()
# print(f"停用词列表:{stopwords}")
print(f"停用词数量:{len(stopwords)}")
# 4.创建TF-IDF向量化器对象
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
# 5.使用TF-IDF向量化器对象进行训练和向量化, fit_transform
words_features = tfidf_vectorizer.fit_transform(words)
print(words_features.shape)
# print(words_features[0])
# print(tfidf_vectorizer.vocabulary_)
# 3.训练随机森林模型
# 1.拆分数据集为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(words_features, labels, test_size=0.2, random_state=5)
print(f"训练集形状:{x_train.shape}")
print(f"测试集形状:{x_test.shape}")
print(f"训练集label形状:{y_train.shape}")
print(f"测试集label形状:{y_test.shape}")
# 创建随机森林模型对象
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(x_train, y_train)
# 4.模型评估
y_pred = rf_model.predict(x_test)
print(f"预测结果y_pred.shape：{y_pred.shape}")
# 计算acc = 预测正确的数量 / 总样本量
acc = accuracy_score(y_test, y_pred)
# 解释 macro / micro / weighted
# macro(更加考虑小类别): 宏平均，计算各个类别指标，然后直接算平均值，每个类别权重一样
# weighted(用的最多): 加权平均，计算各个类别指标，然后根据每个类别样本数量进行加权平均，每个类别权重不一样
# micro(用的很少): 微平均，汇总每个类别的 TP/FP/FN,然后统一计算指标，大类权重更大
# 计算precision = TP / (TP+FP)
precision = precision_score(y_test, y_pred, average='macro')
print(f"precision:{precision}")
# 计算recall = TP / (TP+FN)
recall = recall_score(y_test, y_pred, average='macro')
print(f"recall:{recall}")
# 计算F1-score = 2*p*r/(p+r)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"f1:{f1}")
# 分类评估报告
print(classification_report(y_test, y_pred))
# 混淆矩阵
print(confusion_matrix(y_test, y_pred))

# 5.保存模型,保存 随机森林模型 和 TF-IDF向量化器 为 pickle文件
# 保存随机森林
with open(config.rf_model_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"保存随机森林模型为：{config.rf_model_path}")
# 保存TF-IDF向量化器
with open(config.tfidf_model_path, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"保存TF-IDF向量化器为：{config.tfidf_model_path}")