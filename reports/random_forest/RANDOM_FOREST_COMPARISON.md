# RANDOM_FOREST_COMPARISON

## 1. Historical Course Artifacts

These rows are course reproduction references, not a controlled fairness comparison.

- `existing_course_artifact_test`: loads `data/model/tfidf_model.pkl` and `rf_model.pkl`, then random forest evaluation code evaluates on old `process_test.csv`.
- `course_logic_reproduction`: reproduces the course training logic on the first 20,000 rows of old `process_train.csv`; TF-IDF is fit before the 16,000/4,000 internal split, then evaluated on old `process_test.csv`.
- Original `random_forest/rf_test.py` actually reads `process_dev.csv`; existing random forest evaluation reads `process_test.csv`, so it is not the same original execution path.

| experiment | eval data | RF train rows | TF-IDF leakage | Accuracy | Macro-F1 | Weighted-F1 | RF size bytes |
|---|---|---:|---|---:|---:|---:|---:|
| existing_course_artifact_test | old process_test.csv | original artifact | not auditable from artifact | 0.739039 | 0.747513 | 0.747487 | 178554208 |
| course_logic_reproduction | old process_test.csv | 16000 | yes, internal validation vocabulary leakage | 0.741341 | 0.749283 | 0.749255 | 178077977 |

## 2. Strict Controlled Comparison

Only this section is a fair controlled comparison for TF-IDF vocabulary leakage. Both experiments use the same v2 source data, same 20,000 sampled rows, same 16,000 train_core rows, same 4,000 internal_holdout rows, same RF parameters, and same dev/test sets. The only intended difference is TF-IDF fit scope.

| experiment | data source | sampled rows | RF train rows | TF-IDF fit rows | RF params | Holdout Macro-F1 | Dev Macro-F1 | Test Macro-F1 | vocabulary size |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| controlled_leaky_20k | v2 process_train_v2.csv | 20000 | 16000 | 20000 | n_estimators=100, random_state=42, n_jobs=-1 | 0.741298 | 0.728830 | 0.741045 | 35284 |
| controlled_clean_20k | v2 process_train_v2.csv | 20000 | 16000 | 16000 | n_estimators=100, random_state=42, n_jobs=-1 | 0.734776 | 0.725742 | 0.738978 | 30924 |

Deltas, leaky minus clean:

- Holdout Macro-F1: 0.006522
- Dev Macro-F1: 0.003088
- Test Macro-F1: 0.002067

## 3. Lightweight Engineering Model

`v2_compact_full` is a separate full-v2 lightweight engineering model, not a controlled fair comparison against the course baseline.

| experiment | train rows | Accuracy | Macro-F1 | RF size bytes | trees | max_depth parameter | model.fit seconds | avg inference ms |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| v2_compact_full | 178707 | 0.685966 | 0.706250 | 1511001 | 50 | 40 | 3.232536 | 0.008824 |

## Model Version Compatibility

The existing course artifact emitted `InconsistentVersionWarning` when loaded: it was saved with scikit-learn 1.7.2 and evaluated here under scikit-learn 1.9.0. The recalculated metrics are internally consistent in this environment, but that is not a guarantee of compatibility in all environments. Best reproduction for the original artifact should use scikit-learn 1.7.2. newly generated random forest models were created under the current 1.9.0 environment. Do not load and re-dump old course artifacts over the original files.

## Source Reference Hashes

`source_reference_hashes.json` records current-state reference hashes for original `random_forest` source files. It cannot retroactively prove historical pre-Day-2 state; from now on it can be used for integrity checks.
