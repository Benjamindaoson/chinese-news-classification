"""
演示
    fasttext文本分类训练，按词分词 + 手动调参
涉及到的API:
    fasttext.train_supervised()
"""

# 导包
import fasttext # 需要安装fasttext: pip install fasttext-wheel
from config import Config

# 1.创建配置对象
config = Config()

# 2.模型训练
model = fasttext.train_supervised(
    input = config.word_train_path, # 按词分词的训练集
    dim=10, # 词向量的维度
    wordNgrams=2,   # N-gram特征的N
    epoch=10,    # 训练轮数
    lr=0.1, # 学习率
)

# 3.模型保存
# ft_model.bin -> ft_model_word_1.bin
model.save_model(config.ft_model_path.replace(".bin", "_word_1.bin"))
print(f'模型保存到: {config.ft_model_path.replace(".bin", "_word_1.bin")}')
# 4.模型评估
result = model.test(config.word_test_path)
print(f"评估结果(样本数，精确率，召回率): {result}")
# 精确率result[1],召回率result[2]
f1_score = 2 * result[1]*result[2]/(result[1]+result[2])
print(f"F1-score: {f1_score}")
# 5.打印关键信息，比如词向量，类别标签
print(f"词向量: {model.get_word_vector('上')}")
print(f"类别标签: {model.get_labels()}")
print(f"词表大小：{len(model.get_words())}")
# print(f"词表：{model.get_words()}")