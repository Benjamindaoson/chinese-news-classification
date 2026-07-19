# BASELINE REPORT

## Experiment Names

- `existing_course_artifact_test`: random forest evaluation code loads course-provided `tfidf_model.pkl` and `rf_model.pkl`, then evaluates on old `process_test.csv`.
- `course_logic_reproduction`: safe reproduction of `rf_train.py` logic into `artifacts/random_forest/baseline`, not into `data/model`.

## Source Logic

- `data_process.py` uses jieba tokenization: `jieba.lcut(text)[:20]` for `words` and full token count for `seq_len`.
- `process_*.csv` fields are `text,label,words,seq_len`.
- `rf_train.py` reads only the first 20,000 rows of `process_train.csv`.
- It fits TF-IDF on all 20,000 rows before `train_test_split(..., test_size=0.2, random_state=5)`.
- The resulting RF training subset is 16,000 rows.
- `process_dev.csv` is not used by `rf_train.py`; original `rf_test.py` reads `process_dev.csv` despite comments saying validation/test inconsistently.

## Leakage

`course_logic_reproduction` has `validation_vocabulary_leakage=true`: the internal 4,000 holdout labels are not used for RF fitting, but their vocabulary and IDF statistics are visible to TF-IDF before the split.

## Metrics

| experiment | eval data | accuracy | macro_f1 | weighted_f1 |
|---|---|---:|---:|---:|
| existing_course_artifact_test | process_test.csv | 0.739039 | 0.747513 | 0.747487 |
| course_logic_reproduction | process_test.csv | 0.741341 | 0.749283 | 0.749255 |

## Model Version Compatibility

The course artifact was saved with scikit-learn 1.7.2 and loaded under 1.9.0, producing `InconsistentVersionWarning`. This report records the metrics recalculated in the current environment, but the safest environment for reproducing the original artifact is scikit-learn 1.7.2. The original model files were not rewritten.

## Integrity

Current reference source hashes are stored in `reports/random_forest/source_reference_hashes.json` under `current_reference_sha256`.
