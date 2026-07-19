# 中文新闻分类实验项目

这是一个中文新闻标题分类项目，围绕数据清洗、EDA、特征工程、随机森林建模、实验对照和结果分析构建。项目目标不是只跑出一个模型分数，而是把一个文本分类任务做成可复现、可审计、可对比的机器学习实验流程。

当前已完成：

- 数据质量核验与 EDA；
- v2 高质量数据集构建；
- 随机森林文本分类基线；
- FastText 字符级/词级文本分类基线；
- BERT 微调模型加载、评估、预测和 API 验证；
- LLM 提示词分类、DeepSeek API 和 Flask 路由验证；
- BERT 动态量化、非结构化剪枝和学生模型蒸馏验证；
- TF-IDF 词表泄漏受控实验；
- 轻量化随机森林模型产物；
- 实验报告和可复现脚本。

## 项目亮点

### 1. 保留原始数据，单独构建高质量 v2 数据集

项目没有直接覆盖原始数据，而是额外生成：

```text
data/v2/train_v2.txt
data/v2/dev_v2.txt
data/v2/test_v2.txt
```

v2 数据集处理了两类常见但容易被忽略的数据问题：

- 同一文本跨 train/dev/test 重复；
- 同一文本对应多个不同标签。

清洗策略包括：

- Unicode NFKC 规范化；
- 首尾空白清理；
- 连续空白压缩；
- 标签冲突样本移除；
- 跨集合重复样本移除；
- 保留原始文本内容，不做过度改写。

v2 数据结果：

| split | 原始行数 | v2行数 | 删除数 |
|---|---:|---:|---:|
| train | 179000 | 178707 | 293 |
| dev | 9993 | 9981 | 12 |
| test | 9990 | 9983 | 7 |

共移除 312 条记录：

- 标签冲突：111 条；
- 跨集合重复：201 条；
- v2 的 train/dev/test 交集均为 0；
- 三个集合仍保留完整 10 类标签。

### 2. 从数据层面控制实验泄漏

项目明确区分训练集、验证集和测试集，避免把验证集或测试集信息提前暴露给模型。

随机森林 v2 实验中，TF-IDF 的正确流程是：

```python
x_train = vectorizer.fit_transform(train_words)
x_dev = vectorizer.transform(dev_words)
x_test = vectorizer.transform(test_words)
```

关键点：

- `fit_transform()` 只能用于训练集；
- dev/test 只能使用训练集拟合好的词表做 `transform()`；
- 否则验证集或测试集词表会泄漏进特征工程阶段，导致指标虚高。

### 3. 用受控实验量化 TF-IDF 泄漏影响

项目不是只判断“是否有问题”，而是通过同数据、同采样、同模型参数的方式，定量比较 leaky 和 clean 两种流程。

受控实验结果：

| 指标 | leaky - clean |
|---|---:|
| Holdout Macro-F1 | +0.006522 |
| Dev Macro-F1 | +0.003088 |
| Test Macro-F1 | +0.002067 |

结论：

> TF-IDF 先拟合全量数据再切分，会带来小幅但可测量的指标抬升。

这部分体现了机器学习实验中非常重要的能力：不是只追求分数，而是判断分数是否可信。

### 4. 同时关注效果、速度和模型体积

项目中训练了一个轻量化随机森林模型：

| 训练集 | Accuracy | Macro-F1 | 模型大小 |
|---:|---:|---:|---:|
| 178707 | 0.685966 | 0.706250 | 1511001 bytes |

这个模型不是为了追求最高分，而是展示工程权衡：

- 模型体积约 1.5MB；
- 推理速度快；
- 依赖简单；
- 适合作为轻量级文本分类基线。

### 5. 保持原始流程，修复 Windows 中文路径运行问题

FastText 的 Windows 原生库无法直接读取包含中文的文件路径。项目保留原训练逻辑，只在 FastText 读写文件的边界增加 ASCII 临时路径兼容层，使当前中文目录下也能完成训练、评估、保存和预测。

当前 FastText 原始数据实验结果：

| 实验 | Test F1 |
|---|---:|
| char_1_default | 0.919520 |
| char_2_auto | 0.911211 |
| word_1_default | 0.907407 |
| word_2_auto | 0.915215 |

### 6. BERT 微调模型复现

项目使用本地 `bert-base-chinese` 预训练权重和已训练好的 BERT 分类权重进行文本分类。当前已验证模型加载、测试集评估、单条预测、一批次训练反向传播和 Flask API 路由。

当前 BERT 测试集评估结果：

| Test Loss | Test Accuracy | Test Macro-F1 |
|---:|---:|---:|
| 0.189667 | 0.941542 | 0.941550 |

### 7. LLM 提示词分类复现

项目使用 OpenAI 兼容 SDK 调用 DeepSeek API，通过系统提示词约束模型只输出 10 个新闻类别之一。当前已验证依赖导入、环境变量读取、预测函数、Flask API 包装和真实 DeepSeek 调用。

当前验证样例：

| 输入文本 | 返回类别 |
|---|---|
| 涉股理财品PK基金正规军 | stocks / finance |

说明：LLM API 输出存在非确定性，同一文本在不同调用中可能返回相近但不同的类别。当前阶段保持原提示词和原调用参数，不做重构。

### 8. BERT 模型压缩复现

项目验证了三类模型压缩方法：动态量化、非结构化剪枝和知识蒸馏。当前阶段保持原始脚本逻辑，只修复蒸馏 API 测试脚本的路由地址。

| 方法 | Test Macro-F1 | 说明 |
|---|---:|---|
| 原始 BERT | 0.941550 | 教师模型 |
| 动态量化 | 0.929600 | 量化 `nn.Linear`，保存为 `bert_quantization.pt` |
| 非结构化剪枝 | 0.940831 | query 权重剪枝比例 0.6，稀疏度约 0.6 |
| 学生模型蒸馏 | 0.896735 | 2 层小 BERT，保存为 `student_model.pt` |

## 涉及知识点

### 数据处理

- 文本分类数据格式校验；
- 空行、异常行、空标签检查；
- 标签合法性检查；
- 内部重复与跨集合重复检查；
- Unicode 文本规范化；
- 数据清洗过程可审计；
- 原始数据哈希保护。

### EDA

- 数据集规模统计；
- 标签分布统计；
- 文本长度统计；
- 类别均衡性分析；
- 清洗前后数据对比。

### 中文 NLP

- 中文新闻标题分类；
- jieba 分词；
- 停用词处理；
- TF-IDF 特征提取；
- BERT Tokenizer 编码；
- 预训练 BERT 下游微调；
- LLM 提示词分类；
- OpenAI 兼容 Chat Completions API；
- PyTorch 动态量化；
- 非结构化剪枝；
- 软标签知识蒸馏；
- 词表拟合范围控制；
- 稀疏矩阵特征建模。

### 机器学习建模

- RandomForestClassifier；
- PyTorch Dataset / DataLoader / collate_fn；
- AdamW 优化器；
- CrossEntropyLoss；
- train/dev/test 三集合实验设计；
- Accuracy、Macro-F1、Weighted-F1 指标；
- 模型保存与加载；
- 单条新闻预测；
- 推理耗时统计。

### 实验方法

- 基线实验；
- v2 数据修复实验；
- 受控变量实验；
- 数据泄漏验证；
- 模型体积与性能权衡；
- 实验结果报告化。

## 项目目录

```text
data/                  数据、清洗脚本、EDA脚本
data/v2/               v2数据集与清洗报告
random_forest/                    随机森林文本分类实验
random_forest/v2/                 v2随机森林实验入口
random_forest/controlled/         TF-IDF泄漏受控实验
fasttext/              FastText文本分类实验
bert_finetuning/                  BERT微调、预测和API实验
llm_prompting/                   LLM提示词分类和API实验
bert_compression/          BERT教师/学生蒸馏和API实验
reports/random_forest/            随机森林实验报告
artifacts/random_forest/          随机森林可上传的小体积实验产物
```

## 当前报告

```text
DATA_BASELINE_REPORT.md
data/v2/DATASET_V2.md
data/v2/data_cleaning_report.json
reports/random_forest/RANDOM_FOREST_COMPARISON.md
reports/random_forest/verification.json
reports/random_forest/source_reference_hashes.json
```

## 环境

当前实验环境：

```text
Python 3.12.10
pandas==3.0.3
numpy==2.4.6
jieba==0.42.1
scikit-learn==1.9.0
joblib==1.5.3
matplotlib==3.10.9
seaborn==0.13.2
torch==2.13.0
transformers==4.57.6
tqdm==4.67.3
openai==2.45.0
python-dotenv==1.2.2
```

安装依赖：

```powershell
pip install -r requirements-random-forest-fasttext.txt
pip install -r requirements-bert.txt
pip install -r requirements-llm.txt
pip install -r requirements-compression.txt
```

## 常用命令

生成 v2 数据集：

```powershell
python data\build_dataset_v2.py
```

运行随机森林实验：

```powershell
python random_forest\run_random_forest.py
```

运行 v2 随机森林实验：

```powershell
python random_forest\v2\run_random_forest_v2.py
```

运行 TF-IDF 泄漏受控实验：

```powershell
python random_forest\controlled\controlled_tfidf_experiment.py
```

生成 FastText 输入数据：

```powershell
python fasttext\data_process.py
```

运行 FastText 字符级手动调参：

```powershell
python fasttext\fasttext_char_1_default.py
```

运行 FastText 词级手动调参：

```powershell
python fasttext\fasttext_word_1_default.py
```

运行 FastText 字符级自动调参：

```powershell
python fasttext\fasttext_char_2_auto.py
```

运行 FastText 词级自动调参：

```powershell
python fasttext\fasttext_word_2_auto.py
```

运行 BERT 离线预测：

```powershell
python bert_finetuning\bert_predict_fun.py
```

运行 BERT 训练脚本：

```powershell
python bert_finetuning\bert_train.py
```

说明：当前配置使用 CPU，完整 BERT 训练耗时较长；本地已通过现有权重完成测试集评估和预测链路验证。

运行 DeepSeek 提示词分类：

```powershell
python llm_prompting\deepseek_predict_fun.py
```

运行 LLM Flask API：

```powershell
python llm_prompting\api_flask_server.py
```

说明：`llm_prompting/.env` 需要配置本地 API key，且该文件不会上传到 GitHub。当前 DeepSeek 调用已验证通过；Qwen 扩展脚本当前返回 401，需更新有效的 `DASHSCOPE_API_KEY` 后再运行。

运行 BERT 动态量化：

```powershell
python bert_finetuning\bert_quantization.py
```

运行 BERT 非结构化剪枝：

```powershell
python bert_finetuning\bert_pruning.py
```

运行教师/学生模型预测：

```powershell
python bert_compression\bert_predict_fun.py
```

运行蒸馏训练脚本：

```powershell
python bert_compression\student_train.py
```

说明：当前 CPU 环境下完整 BERT/蒸馏训练耗时较长；本地已通过已有权重完成评估、预测、API 和一批次反向传播验证。
