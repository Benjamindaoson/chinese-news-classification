# V2 REPORT

## Experiment Name

`v2_compact_full` uses all 178,707 rows of `train_v2` and evaluates on `test_v2`. It is a lightweight full-data engineering model, not a fair controlled comparison with the course baseline.

## Responsibilities

- `rf_train_v2.py`: trains and writes `artifacts/day02/v2/tfidf_v2.pkl` and `rf_v2.pkl`.
- `rf_evaluate_v2.py`: read-only evaluator; it loads saved v2 artifacts, transforms dev/test, predicts, writes `rf_evaluate_v2_readonly.json`, and verifies model hashes are unchanged.

## Strict TF-IDF Flow

TF-IDF is fit only on `train_v2`; `dev_v2` and `test_v2` are transform-only. `test_v2` is not used for parameter selection.

## Selected Compact Parameters

`{'n_estimators': 50, 'max_depth': 40, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1}`

## Results

| split | accuracy | macro_f1 | weighted_f1 |
|---|---:|---:|---:|
| dev_v2 read-only | 0.671175 | 0.691279 | 0.691299 |
| test_v2 | 0.685966 | 0.706250 | 0.706232 |

## Size and Speed

- TF-IDF vocabulary size: 112395
- RF artifact size bytes: 1511001
- TF-IDF artifact size bytes: 2492048
- training seconds: 3.232536
- average inference ms: 0.008824

## Hash Check

Read-only evaluator model hashes unchanged: True.

## Error Samples

See `reports/day02/v2/error_samples.csv`. RandomForest remains a sparse-feature baseline and should not be described as semantic understanding.
