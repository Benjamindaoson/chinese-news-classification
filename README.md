# 中文新闻分类实验项目

这是一个围绕中文新闻标题分类的课程复现与工程化改造项目。当前项目保留课程原始数据和代码，同时新增一套经过质量升级的数据集与可复现实验，用来区分“课程结果复现”和“更严谨的模型对比”。

当前进度：

- Day 1：课程数据清洗与 EDA 基线复现，已完成。
- Day 1 v2：数据质量升级，已完成。
- Day 2：随机森林课程复现、泄漏验证和 v2 轻量化模型，已完成。
- Day 3：FastText，尚未开始。

## 项目目录

```text
01-data/                  数据、清洗脚本、EDA脚本
01-data/v2/               v2数据集与清洗报告
02-rf/                    随机森林课程代码与Day 2复现实验
02-rf/v2/                 随机森林v2实验入口
02-rf/controlled/         TF-IDF泄漏受控实验
03-fasttext/              FastText课程代码，后续Day 3处理
04-bert/                  BERT课程代码，后续阶段处理
05-llm/                   LLM相关代码，当前未进入
06-bert_distill/          BERT蒸馏相关代码，当前未进入
reports/day02/            Day 2实验报告
artifacts/day02/          Day 2可上传的小体积实验产物
```

## 数据集

项目中保留两套数据。

课程基线数据：

```text
01-data/train.txt
01-data/dev.txt
01-data/test.txt
```

用途：

- 复现课程原始结果；
- 对照课程讲义指标；
- 验证原课程代码行为。

v2升级数据：

```text
01-data/v2/train_v2.txt
01-data/v2/dev_v2.txt
01-data/v2/test_v2.txt
```

用途：

- 后续严谨实验；
- 正式模型对比；
- 项目报告和作品集展示。

v2清洗结果：

| split | 原始行数 | v2行数 | 删除数 |
|---|---:|---:|---:|
| train | 179000 | 178707 | 293 |
| dev | 9993 | 9981 | 12 |
| test | 9990 | 9983 | 7 |

共移除 312 条记录：

- 标签冲突：111 条；
- 跨集合重复：201 条；
- v2 的 train/dev/test 交集均为 0；
- 原始课程数据文件未被覆盖。

## Day 1结果

Day 1完成了课程数据路径、依赖、文件格式、行数、重复行和 EDA 结果核验。

课程 EDA 可复现结果：

- 训练集：179000 条；
- 平均文本长度：约 19.209；
- 最短文本长度：3；
- 最长文本长度：38；
- 10 个类别整体均衡。

报告文件：

```text
DAY01_BASELINE_REPORT.md
01-data/v2/DATASET_V2.md
01-data/v2/data_cleaning_report.json
```

## Day 2结果

Day 2完成了随机森林三类实验。

| 实验 | 用途 |
|---|---|
| existing_course_artifact_test | 了解课程提供模型的水平 |
| course_logic_reproduction | 复现课程原训练逻辑 |
| controlled_leaky_20k / clean_20k | 严格测量 TF-IDF 泄漏影响 |
| v2_compact_full | 展示效果、速度、体积之间的工程权衡 |

核心结论：

- 原课程先拟合 TF-IDF 再切分数据，方法上存在验证集词表泄漏；
- 在这套数据中，泄漏造成了小幅、可测量的指标抬升；
- `v2_compact_full` 不是课程模型的公平性能对照，而是一个约 1.5MB 的轻量化随机森林工程方案。

受控实验结果：

| 指标 | leaky - clean |
|---|---:|
| Holdout Macro-F1 | +0.006522 |
| Dev Macro-F1 | +0.003088 |
| Test Macro-F1 | +0.002067 |

v2轻量化随机森林结果：

| 训练集 | Accuracy | Macro-F1 | 模型大小 |
|---:|---:|---:|---:|
| 178707 | 0.685966 | 0.706250 | 1511001 bytes |

报告文件：

```text
reports/day02/DAY02_COMPARISON.md
reports/day02/verification.json
reports/day02/source_reference_hashes.json
```

## 环境

Day 2当前环境记录：

```text
Python 3.12.10
pandas==3.0.3
numpy==2.4.6
jieba==0.42.1
scikit-learn==1.9.0
joblib==1.5.3
matplotlib==3.10.9
seaborn==0.13.2
```

安装依赖：

```powershell
pip install -r requirements-day02.txt
```

如果只复现课程原始模型 artifact，参考：

```powershell
pip install -r requirements-day02-course-artifact.txt
```

## 常用命令

复现 Day 1 v2 数据清洗：

```powershell
python 01-data\build_dataset_v2.py
```

运行 Day 2随机森林总入口：

```powershell
python 02-rf\run_day02.py
```

运行 v2随机森林实验：

```powershell
python 02-rf\v2\run_day02_v2.py
```

运行 TF-IDF泄漏受控实验：

```powershell
python 02-rf\controlled\controlled_tfidf_experiment.py
```

## GitHub上传说明

仓库已排除不适合普通 GitHub 上传的内容：

- `.env`
- `.venv/`
- `.idea/`
- `__pycache__/`
- 原课程大模型目录 `01-data/model/`
- BERT预训练模型目录 `01-data/bert-base-chinese/`
- 超过 GitHub 100MB 限制的随机森林大模型产物

已保留可上传的小体积代码、报告、v2数据和轻量化实验产物。

## 下一步

建议顺序：

```text
手写最小Day 2
-> 对照当前Day 2代码检查理解
-> 再进入Day 3 FastText
```

Day 3开始前，不建议继续修改 Day 1 和 Day 2 数据结论。
