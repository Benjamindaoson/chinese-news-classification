# Chinanews 百万级中文新闻分类实验

本目录用于把仓库现有的中文新闻分类实验扩展到 **Chinanews** 百万级数据规模，并保持“数据隔离、可复现训练、固定测试集、模型压缩”的实验原则。

## 数据集

Chinanews 收录 2008–2016 年中国新闻网新闻，包含 **7 个类别**，公开实验设置为：

- 训练数据：1,400,000 条
- 官方测试数据：112,000 条
- 总规模：1,512,000 条
- 各类别样本数均衡
- 单条样本使用新闻首段文本

仓库不直接提交原始数据文件。请自行取得公开数据后，将 `train.csv` 与 `test.csv` 放到本地目录，例如：

```text
external_data/chinanews/
├── train.csv
└── test.csv
```

`prepare_dataset.py` 默认兼容无表头 CSV：第一列为类别编号，其余文本列会拼接为分类输入。如果你的数据格式不同，可通过命令行指定分隔符和类别列。

## 数据划分原则

官方数据只有训练集与测试集。本项目从官方训练数据中按类别分层划出验证集，并保持官方测试集仅用于最终评测：

```text
官方 1,400,000 条训练数据
    ├── 训练集：约 95%
    └── 验证集：约 5%

官方 112,000 条测试数据
    └── 固定测试集：不参与调参、早停和模型选择
```

这样可以避免把测试集反复用于超参数选择造成评测泄漏。

## 1. 数据准备

```powershell
python chinanews\prepare_dataset.py ^
  --train_csv external_data\chinanews\train.csv ^
  --test_csv external_data\chinanews\test.csv ^
  --output_dir data\chinanews ^
  --validation_ratio 0.05
```

输出：

```text
data/chinanews/
├── train.tsv
├── validation.tsv
├── test.tsv
├── labels.json
└── dataset_report.json
```

脚本会执行：

- Unicode NFKC 文本规范化；
- 空文本与非法标签检查；
- 训练数据内部去重；
- 官方测试数据内部去重；
- 训练 / 验证 / 测试跨集合重复检查；
- 按类别分层划分验证集；
- 输出类别分布和数据规模报告。

## 2. BERT 分类训练

```powershell
python chinanews\train_bert.py ^
  --data_dir data\chinanews ^
  --model_name bert-base-chinese ^
  --output_dir outputs\chinanews-bert ^
  --epochs 3 ^
  --learning_rate 2e-5 ^
  --train_batch_size 32 ^
  --eval_batch_size 64 ^
  --max_length 256 ^
  --gradient_accumulation_steps 2 ^
  --fp16
```

训练策略：

- `bert-base-chinese` 预训练模型；
- 多分类交叉熵损失；
- AdamW 优化；
- 学习率预热；
- 动态填充；
- 混合精度训练；
- 梯度累积；
- 基于验证集宏平均 F1 选择最佳检查点；
- 官方测试集仅在模型选择完成后做最终评测。

输出指标：

- 准确率；
- 宏平均 F1；
- 加权 F1；
- 各类别精确率 / 召回率 / F1；
- 混淆矩阵；
- 训练与验证过程指标。

## 3. TextCNN / BiLSTM 基线

```powershell
python chinanews\train_neural_baselines.py --model textcnn --data_dir data\chinanews
python chinanews\train_neural_baselines.py --model bilstm --data_dir data\chinanews
```

两个模型使用相同的数据划分和固定测试集，避免模型之间因数据处理差异造成不可比。

## 4. 模型压缩

仓库现有 `bert_compression/` 已包含三类实现：

1. **INT8 动态量化**：对 `nn.Linear` 层进行动态量化；
2. **全局 L1 非结构化剪枝**：对各编码器层自注意力模块的 Query 投影矩阵进行全局剪枝；
3. **知识蒸馏**：学生模型联合优化真实标签交叉熵和教师软输出的 KL 散度。

百万级实验完成 BERT 教师模型训练后，可复用上述压缩思路，但应重新生成对应检查点并在 Chinanews 固定测试集上重新评测。

## 结果治理

**本目录不预填任何尚未实际运行的百万级指标。**

仓库根目录 README 中现有 BERT / 量化 / 剪枝 / 蒸馏指标来自原 10 类约 19.9 万条数据实验；Chinanews 151.2 万条实验必须重新训练后再写入正式结果，避免数据集与指标来源不一致。

建议最终形成：

```text
FastText
   ↓
TextCNN / BiLSTM
   ↓
BERT Fine-tuning
   ↓
错误分析 / 混淆矩阵
   ↓
INT8 动态量化
   ↓
非结构化剪枝
   ↓
知识蒸馏
```

以统一固定测试集比较分类效果、模型规模与推理效率。