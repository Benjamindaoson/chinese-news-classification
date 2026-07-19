# CONTROLLED_EXPERIMENT_REPORT

## Question

在相同v2数据、相同20000条样本、相同16000条RF训练样本、相同4000条内部holdout、相同RF参数和相同评估集下，只改变TF-IDF fit范围，观察提前看到holdout词表与IDF对指标的影响。

## Controlled Design

- sample source: `data/v2/process_train_v2.csv`
- sample size: 20000
- sampler: `StratifiedShuffleSplit(random_state=42)`
- split: 16000 train_core / 4000 internal_holdout, `random_state=5`
- RF params: `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`
- leaky TF-IDF fit rows: 20000
- clean TF-IDF fit rows: 16000

## Results

| experiment | holdout macro_f1 | dev macro_f1 | test macro_f1 | vocabulary size |
|---|---:|---:|---:|---:|
| controlled_leaky_20k | 0.741298 | 0.728830 | 0.741045 | 35284 |
| controlled_clean_20k | 0.734776 | 0.725742 | 0.738978 | 30924 |

## Deltas: leaky - clean

- holdout macro_f1: 0.006522
- dev macro_f1: 0.003088
- test macro_f1: 0.002067

## Vocabulary

- leaky vocabulary size: 35284
- clean vocabulary size: 30924
- vocabulary intersection: 30924
- only in leaky vocabulary: 4360

## Boundary

This experiment only isolates TF-IDF validation vocabulary leakage inside the same v2 sample. It does not prove that v2 is better than the original data, does not quantify the full score impact of Day 1 cleaning, and does not claim every course metric is invalid.
