> **STATUS: MAINTAINED LEGACY CASE STUDY**
>
> **LEGACY MODEL-SYSTEMS CASE STUDY**
>
> This repository is retained for reproducibility of earlier NLP fine-tuning / compression work.
> It is not a current flagship project and is no longer the primary representation of the author's LLM/model-systems work.
> Current flagship model work is listed on the GitHub profile.

# 中文新闻分类与 BERT 模型压缩

一个面向中文文本分类的可复现深度学习实验项目，覆盖 **数据治理 → 多模型基线 → BERT 分类 → 错误分析 → 模型压缩 → 推理服务**。

项目目前保留两条实验线：

1. **已验证实验线**：原 10 类、约 19.9 万条中文新闻数据，已有可复现实验结果；
2. **百万级扩展实验线**：Chinanews 151.2 万条、7 类中文新闻数据，已加入数据准备、TextCNN / BiLSTM 基线和 BERT 大规模训练代码，百万级结果需要在 GPU 环境重新训练后写入。

> 仓库不会把旧数据集上的指标直接迁移到 Chinanews。不同数据集的结果必须分别训练、分别评测。

## 项目亮点

- 数据清洗、标签冲突检测、跨训练集 / 验证集 / 测试集重复治理；
- FastText、随机森林、TextCNN、BiLSTM、BERT 多种分类基线；
- `bert-base-chinese` 下游分类训练、评估与预测；
- 宏平均 F1、加权 F1、分类别指标和混淆矩阵；
- INT8 动态量化、全局 L1 非结构化剪枝、知识蒸馏；
- 固定测试集、验证集模型选择、数据泄漏受控实验；
- Flask / Streamlit 推理链路；
- Chinanews 151.2 万条数据的大规模训练入口。

## 一、百万级 Chinanews 实验线

### 数据规模

Chinanews 收录 2008–2016 年中国新闻网新闻，包含 **7 个类别**：

- 官方训练数据：**1,400,000 条**；
- 官方测试数据：**112,000 条**；
- 总规模：**1,512,000 条**；
- 各类别样本量均衡；
- 单条样本使用新闻首段文本。

官方数据只提供训练集与测试集，因此本项目从官方训练数据中使用**确定性哈希划分**生成约 5% 的验证集，并检查各类别分布是否保持基本一致；官方测试集始终仅用于最终评测。

```text
官方训练集 1,400,000
├── 训练集：约 95%
└── 验证集：约 5%

官方测试集 112,000
└── 固定测试集：不参与调参、早停和模型选择
```

### 数据准备

原始数据不直接提交到 GitHub。将本地 `train.csv` 与 `test.csv` 准备好后运行：

```powershell
pip install -r requirements-chinanews.txt

python chinanews\prepare_dataset.py ^
  --train_csv external_data\chinanews\train.csv ^
  --test_csv external_data\chinanews\test.csv ^
  --output_dir data\chinanews ^
  --validation_ratio 0.05
```

数据准备脚本会执行：

- Unicode NFKC 文本规范化；
- 空文本及非法标签检查；
- 训练数据内部去重；
- 测试数据内部去重；
- 训练 / 验证 / 测试跨集合重复检查；
- 确定性哈希划分验证集；
- 类别分布和数据规模报告输出。

输出：

```text
data/chinanews/
├── train.tsv
├── validation.tsv
├── test.tsv
├── labels.json
└── dataset_report.json
```

### TextCNN / BiLSTM 基线

两个神经网络基线与 BERT 使用同一数据划分和固定测试集：

```powershell
python chinanews\train_neural_baselines.py --model textcnn --data_dir data\chinanews
python chinanews\train_neural_baselines.py --model bilstm --data_dir data\chinanews
```

其中：

- **TextCNN**：词元嵌入 + 多尺度一维卷积 + 最大池化 + 分类层；
- **BiLSTM**：词元嵌入 + 双向长短期记忆网络 + 掩码平均池化 + 分类层。

### BERT 大规模分类训练

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

训练流程包括：

- 分词与编码；
- 多分类交叉熵损失；
- AdamW 优化；
- 学习率预热；
- 动态填充；
- 混合精度训练；
- 梯度累积；
- 按长度分组降低填充开销；
- 基于验证集宏平均 F1 选择最佳检查点；
- 训练完成后才在固定测试集上进行最终评测。

最终输出：

- 准确率；
- 宏平均 F1；
- 加权 F1；
- 分类别精确率 / 召回率 / F1；
- 混淆矩阵；
- 最佳模型检查点；
- 最终测试结果 JSON。

### 百万级结果状态

**当前仓库尚未提交 Chinanews 151.2 万条数据上的正式 GPU 实验结果。**

这部分刻意不预填任何推测指标。完成百万级训练后，应将真实结果补充到本 README，并重新执行量化、剪枝和蒸馏实验。

详细说明见：[`chinanews/README.md`](chinanews/README.md)。

## 二、已验证的原始 10 类实验线

### 数据治理

原始数据规模约 19.9 万条，历史划分为：

| 数据划分 | 原始样本数 | 清洗后样本数 |
|---|---:|---:|
| 训练集 | 179,000 | 178,707 |
| 验证集 | 9,993 | 9,981 |
| 测试集 | 9,990 | 9,983 |

共移除 312 条冲突或跨集合重复记录：

- 标签冲突：111 条；
- 跨集合重复：201 条；
- 清洗后的训练 / 验证 / 测试交集为 0；
- 仍保留完整 10 类标签。

清洗策略：

- Unicode NFKC 规范化；
- 首尾空白清理；
- 连续空白压缩；
- 标签冲突样本移除；
- 跨集合重复样本移除；
- 原始数据保留，另行生成 v2 数据集。

### TF-IDF 数据泄漏受控实验

项目对比“先拟合全量数据再切分”和“只在训练集拟合词表”的差异：

| 指标 | 泄漏流程 - 正确流程 |
|---|---:|
| 留出集宏平均 F1 | +0.006522 |
| 验证集宏平均 F1 | +0.003088 |
| 测试集宏平均 F1 | +0.002067 |

结论：将验证集 / 测试集词表信息提前暴露给 TF-IDF 会带来小幅但可测量的指标虚高。

### FastText 基线

| 实验 | 测试集 F1 |
|---|---:|
| 字符级默认参数 | 0.919520 |
| 字符级自动调参 | 0.911211 |
| 词级默认参数 | 0.907407 |
| 词级自动调参 | 0.915215 |

### BERT 分类结果

项目使用本地 `bert-base-chinese` 预训练权重和已训练分类权重，已验证模型加载、测试集评估、单条预测、训练反向传播和 Flask API 路由。

| 测试损失 | 测试准确率 | 测试宏平均 F1 |
|---:|---:|---:|
| 0.189667 | 0.941542 | 0.941550 |

## 三、BERT 模型压缩

现有压缩代码均可以直接检查，术语与实现保持一致。

### 1. INT8 动态量化

`bert_compression/bert_quantization.py` 使用 PyTorch 动态量化接口：

```text
目标模块：nn.Linear
量化类型：qint8
运行设备：CPU
```

历史测试集宏平均 F1：**0.929600**。

### 2. 全局 L1 非结构化剪枝

`bert_compression/bert_pruning.py` 对各 BERT 编码器层自注意力模块中的 **Query 投影矩阵权重**执行全局 L1 非结构化剪枝：

```text
prune.global_unstructured
pruning_method = L1Unstructured
amount = 0.6
```

即剪枝比例约 **60%**，历史测试集宏平均 F1：**0.940831**。

需要注意：非结构化剪枝主要制造权重稀疏性，本身不等于结构化加速，也不会自动减少模型文件体积。

### 3. 知识蒸馏

学生模型采用 **2 层 BERT 结构**：

- 隐藏维度：256；
- 编码器层数：2；
- 注意力头数：8。

蒸馏损失由两部分组成：

```text
真实标签交叉熵损失
+
教师软输出与学生输出之间的 KL 散度
```

并使用温度参数进行软标签蒸馏。历史测试集宏平均 F1：**0.896735**。

### 历史压缩结果汇总

| 方法 | 测试集宏平均 F1 | 说明 |
|---|---:|---|
| 原始 BERT | 0.941550 | 教师模型 |
| INT8 动态量化 | 0.929600 | 量化线性层 |
| 全局 L1 非结构化剪枝 | 0.940831 | Query 投影矩阵剪枝比例 60% |
| 2 层学生模型知识蒸馏 | 0.896735 | 交叉熵 + KL 散度蒸馏 |

## 四、项目目录

```text
chinanews/                        Chinanews 151.2 万条大规模实验
data/                             原始 10 类数据、清洗与 EDA
data/v2/                          清洗后的 v2 数据集
random_forest/                    随机森林文本分类实验
random_forest/v2/                 v2 随机森林实验
random_forest/controlled/         TF-IDF 数据泄漏受控实验
fasttext/                         FastText 文本分类实验
bert_finetuning/                  BERT 分类训练、预测与 API
bert_compression/                 BERT 量化、剪枝与知识蒸馏
llm_prompting/                    LLM 提示词分类实验
reports/                          实验报告
artifacts/                        可公开的小体积实验产物
```

## 五、安装

原始实验：

```powershell
pip install -r requirements-random-forest-fasttext.txt
pip install -r requirements-bert.txt
pip install -r requirements-compression.txt
pip install -r requirements-llm.txt
```

Chinanews 百万级实验：

```powershell
pip install -r requirements-chinanews.txt
```

## 六、常用命令

生成原始 v2 数据集：

```powershell
python data\build_dataset_v2.py
```

运行 TF-IDF 泄漏受控实验：

```powershell
python random_forest\controlled\controlled_tfidf_experiment.py
```

运行 BERT 分类训练：

```powershell
python bert_finetuning\bert_train.py
```

运行 BERT 动态量化：

```powershell
python bert_compression\bert_quantization.py
```

运行 BERT 非结构化剪枝：

```powershell
python bert_compression\bert_pruning.py
```

运行知识蒸馏：

```powershell
python bert_compression\student_train.py
```

运行 Chinanews 数据准备与 BERT 训练：

```powershell
python chinanews\prepare_dataset.py --help
python chinanews\train_bert.py --help
```

## 七、实验原则

这个仓库不以单一高分为目标，重点是保证结果可以解释和复现：

1. 数据清洗与模型训练分离；
2. 验证集用于模型选择，固定测试集用于最终评测；
3. 不允许测试集参与词表拟合或超参数调节；
4. 不将不同数据集上的指标混用；
5. 压缩实验使用统一测试集与教师模型作为参照；
6. 百万级实验只有在真实运行后才写入正式指标。
