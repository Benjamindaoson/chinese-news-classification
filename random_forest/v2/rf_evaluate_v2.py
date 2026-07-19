from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[2]
DATA_V2 = ROOT / "data" / "v2"
ARTIFACTS = ROOT / "artifacts" / "random_forest" / "v2"
REPORT = ROOT / "reports" / "random_forest" / "v2" / "rf_evaluate_v2_readonly.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_split(model, vectorizer, path: Path) -> dict[str, float | int | str]:
    df = pd.read_csv(path)
    features = vectorizer.transform(df["words"])
    pred = model.predict(features)
    macro = precision_recall_fscore_support(df["label"], pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(df["label"], pred, average="weighted", zero_division=0)
    return {
        "path": str(path.resolve()),
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(df["label"], pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
    }


def main() -> int:
    tfidf_path = ARTIFACTS / "tfidf_v2.pkl"
    rf_path = ARTIFACTS / "rf_v2.pkl"
    if not tfidf_path.exists() or not rf_path.exists():
        raise FileNotFoundError("missing v2 model artifacts; run rf_train_v2.py first")

    before = {"tfidf_v2.pkl": sha256(tfidf_path), "rf_v2.pkl": sha256(rf_path)}
    vectorizer = joblib.load(tfidf_path)
    model = joblib.load(rf_path)
    result = {
        "mode": "read_only_evaluation",
        "dev": evaluate_split(model, vectorizer, DATA_V2 / "process_dev_v2.csv"),
        "test": evaluate_split(model, vectorizer, DATA_V2 / "process_test_v2.csv"),
    }
    after = {"tfidf_v2.pkl": sha256(tfidf_path), "rf_v2.pkl": sha256(rf_path)}
    result["model_sha256_before"] = before
    result["model_sha256_after"] = after
    result["model_hashes_unchanged"] = before == after
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if before != after:
        raise AssertionError("v2 model hashes changed during read-only evaluation")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
