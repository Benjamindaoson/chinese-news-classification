# RF v2 Scripts

- `rf_train_v2.py`: trains the v2 compact RandomForest model and writes artifacts under `artifacts/random_forest/v2`.
- `rf_evaluate_v2.py`: read-only evaluator. It loads `artifacts/random_forest/v2/tfidf_v2.pkl` and `rf_v2.pkl`, evaluates `process_dev_v2.csv` and `process_test_v2.csv`, and verifies model SHA-256 hashes are unchanged.
- `rf_predict_v2.py`: loads the saved v2 compact model for single-text prediction.

Do not use `rf_evaluate_v2.py` for training.
