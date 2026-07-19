from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_V2 = ROOT / "01-data" / "v2"
REPORT_DIR = ROOT / "reports" / "day02" / "controlled"
ARTIFACT_DIR = ROOT / "artifacts" / "day02" / "controlled"
LABELS = list(range(10))
SAMPLE_SIZE = 20_000
TRAIN_SIZE = 16_000
RANDOM_SAMPLE_STATE = 42
RANDOM_SPLIT_STATE = 5


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_class_names() -> list[str]:
    return [
        line.strip()
        for line in (ROOT / "01-data" / "class.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def read_stopwords() -> list[str]:
    return [
        line.strip()
        for line in (ROOT / "01-data" / "stopwords.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["text", "label", "words", "seq_len"]
    if list(df.columns) != expected:
        raise ValueError(f"{path} columns must be {expected}, got {list(df.columns)}")
    return df


def sample_indices(train_df: pd.DataFrame) -> dict[str, Any]:
    sampler = StratifiedShuffleSplit(n_splits=1, train_size=SAMPLE_SIZE, random_state=RANDOM_SAMPLE_STATE)
    sample_idx, _ = next(sampler.split(train_df, train_df["label"]))
    sample_df = train_df.iloc[sample_idx].reset_index(drop=False).rename(columns={"index": "original_index"})

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=RANDOM_SPLIT_STATE)
    train_pos, holdout_pos = next(splitter.split(sample_df, sample_df["label"]))
    payload = {
        "sample_size": SAMPLE_SIZE,
        "train_core_size": len(train_pos),
        "internal_holdout_size": len(holdout_pos),
        "sample_random_state": RANDOM_SAMPLE_STATE,
        "split_random_state": RANDOM_SPLIT_STATE,
        "sample_original_indices": [int(sample_df.iloc[pos]["original_index"]) for pos in range(len(sample_df))],
        "train_core_positions": [int(pos) for pos in train_pos],
        "internal_holdout_positions": [int(pos) for pos in holdout_pos],
        "sample_label_counts": {str(k): int(v) for k, v in sample_df["label"].value_counts().sort_index().items()},
        "train_core_label_counts": {str(k): int(v) for k, v in sample_df.iloc[train_pos]["label"].value_counts().sort_index().items()},
        "internal_holdout_label_counts": {str(k): int(v) for k, v in sample_df.iloc[holdout_pos]["label"].value_counts().sort_index().items()},
    }
    payload["sample_indices_sha256"] = sha256_bytes(
        json.dumps(payload["sample_original_indices"], separators=(",", ":")).encode("utf-8")
    )
    payload["train_core_indices_sha256"] = sha256_bytes(
        json.dumps(payload["train_core_positions"], separators=(",", ":")).encode("utf-8")
    )
    payload["internal_holdout_indices_sha256"] = sha256_bytes(
        json.dumps(payload["internal_holdout_positions"], separators=(",", ":")).encode("utf-8")
    )
    return {"sample_df": sample_df, "train_pos": train_pos, "holdout_pos": holdout_pos, "payload": payload}


def evaluate(model: Any, vectorizer: Any, df: pd.DataFrame, split_name: str, class_names: list[str]) -> tuple[dict[str, Any], list[int]]:
    features = vectorizer.transform(df["words"])
    start = time.perf_counter()
    pred = model.predict(features)
    elapsed = time.perf_counter() - start
    macro = precision_recall_fscore_support(df["label"], pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(df["label"], pred, average="weighted", zero_division=0)
    per = precision_recall_fscore_support(df["label"], pred, labels=LABELS, zero_division=0)
    per_class = []
    for i, label in enumerate(LABELS):
        per_class.append(
            {
                "label": label,
                "class_name": class_names[label],
                "precision": float(per[0][i]),
                "recall": float(per[1][i]),
                "f1": float(per[2][i]),
                "support": int(per[3][i]),
            }
        )
    metrics = {
        "split": split_name,
        "accuracy": float(accuracy_score(df["label"], pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "total_inference_seconds": elapsed,
        "average_inference_ms": elapsed / len(df) * 1000,
        "samples_per_second": len(df) / elapsed if elapsed else None,
        "per_class": per_class,
    }
    return metrics, [int(x) for x in pred]


def write_classification_report(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_confusion(prefix: str, true: pd.Series, pred: list[int], class_names: list[str]) -> dict[str, Any]:
    cm = confusion_matrix(true, pred, labels=LABELS)
    df = pd.DataFrame(cm, index=[f"{i}:{class_names[i]}" for i in LABELS], columns=[f"{i}:{class_names[i]}" for i in LABELS])
    csv_path = REPORT_DIR / f"controlled_confusion_matrix_{prefix}.csv"
    png_path = REPORT_DIR / f"controlled_confusion_matrix_{prefix}.png"
    df.to_csv(csv_path, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"controlled_{prefix} test confusion matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(10), [str(i) for i in LABELS])
    ax.set_yticks(range(10), [str(i) for i in LABELS])
    for i in LABELS:
        for j in LABELS:
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return {
        "sum": int(cm.sum()),
        "diagonal_sum": int(cm.trace()),
        "accuracy_from_confusion": float(cm.trace() / cm.sum()),
    }


def write_errors(prefix: str, df: pd.DataFrame, pred: list[int], model: Any, vectorizer: Any, class_names: list[str]) -> None:
    errors = df.copy()
    errors["predicted_label"] = pred
    errors = errors[errors["label"] != errors["predicted_label"]].head(200)
    probs = model.predict_proba(vectorizer.transform(errors["words"])) if len(errors) and hasattr(model, "predict_proba") else []
    rows = []
    for idx, (_, row) in enumerate(errors.iterrows()):
        pairs = sorted(enumerate(probs[idx]), key=lambda item: item[1], reverse=True) if len(errors) else []
        rows.append(
            {
                "text": row["text"],
                "true_label": int(row["label"]),
                "true_class_name": class_names[int(row["label"])],
                "predicted_label": int(row["predicted_label"]),
                "predicted_class_name": class_names[int(row["predicted_label"])],
                "text_length": len(str(row["text"])),
                "predicted_probability": float(dict(pairs).get(int(row["predicted_label"]), 0.0)) if pairs else "",
                "top2_label": int(pairs[1][0]) if len(pairs) > 1 else "",
                "top2_probability": float(pairs[1][1]) if len(pairs) > 1 else "",
            }
        )
    pd.DataFrame(rows).to_csv(REPORT_DIR / f"controlled_error_samples_{prefix}.csv", index=False, encoding="utf-8-sig")


def train_one(name: str, sample_df: pd.DataFrame, train_pos: Any, holdout_pos: Any, dev_df: pd.DataFrame, test_df: pd.DataFrame, stopwords: list[str], class_names: list[str]) -> dict[str, Any]:
    start = time.perf_counter()
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    if name == "leaky":
        all_features = vectorizer.fit_transform(sample_df["words"])
        x_train = all_features[train_pos]
        x_holdout = all_features[holdout_pos]
        leakage = True
        fit_rows = len(sample_df)
    else:
        x_train = vectorizer.fit_transform(sample_df.iloc[train_pos]["words"])
        x_holdout = vectorizer.transform(sample_df.iloc[holdout_pos]["words"])
        leakage = False
        fit_rows = len(train_pos)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, sample_df.iloc[train_pos]["label"])
    train_seconds = time.perf_counter() - start

    holdout_df = sample_df.iloc[holdout_pos].reset_index(drop=True)
    holdout_metrics, holdout_pred = evaluate(model, vectorizer, holdout_df, "internal_holdout", class_names)
    dev_metrics, dev_pred = evaluate(model, vectorizer, dev_df, "dev_v2", class_names)
    test_metrics, test_pred = evaluate(model, vectorizer, test_df, "test_v2", class_names)

    joblib.dump(vectorizer, ARTIFACT_DIR / f"{name}_tfidf.pkl")
    joblib.dump(model, ARTIFACT_DIR / f"{name}_rf.pkl")
    reloaded_vectorizer = joblib.load(ARTIFACT_DIR / f"{name}_tfidf.pkl")
    reloaded_model = joblib.load(ARTIFACT_DIR / f"{name}_rf.pkl")
    reload_pred = list(reloaded_model.predict(reloaded_vectorizer.transform(test_df.head(50)["words"])))
    original_pred = list(model.predict(vectorizer.transform(test_df.head(50)["words"])))
    if reload_pred != original_pred:
        raise AssertionError(f"{name} reload prediction mismatch")

    write_classification_report(REPORT_DIR / f"controlled_classification_report_{name}.csv", test_metrics["per_class"])
    cm_info = write_confusion(name, test_df["label"], test_pred, class_names)
    write_errors(name, test_df, test_pred, model, vectorizer, class_names)

    vocab = set(vectorizer.vocabulary_)
    return {
        "experiment": f"controlled_{name}_20k",
        "validation_vocabulary_leakage": leakage,
        "tfidf_fit_rows": fit_rows,
        "rf_train_rows": len(train_pos),
        "internal_holdout_rows": len(holdout_pos),
        "rf_params": model.get_params(),
        "tfidf_params": {key: str(value) for key, value in vectorizer.get_params().items()},
        "vocabulary_size": len(vocab),
        "idf_min": float(vectorizer.idf_.min()),
        "idf_max": float(vectorizer.idf_.max()),
        "idf_mean": float(vectorizer.idf_.mean()),
        "train_matrix_shape": list(x_train.shape),
        "holdout_matrix_shape": list(x_holdout.shape),
        "training_seconds": train_seconds,
        "holdout": holdout_metrics,
        "dev": dev_metrics,
        "test": test_metrics,
        "test_confusion": cm_info,
        "artifacts": {
            "tfidf": {
                "path": str((ARTIFACT_DIR / f"{name}_tfidf.pkl").resolve()),
                "size_bytes": (ARTIFACT_DIR / f"{name}_tfidf.pkl").stat().st_size,
                "sha256": sha256_file(ARTIFACT_DIR / f"{name}_tfidf.pkl"),
            },
            "rf": {
                "path": str((ARTIFACT_DIR / f"{name}_rf.pkl").resolve()),
                "size_bytes": (ARTIFACT_DIR / f"{name}_rf.pkl").stat().st_size,
                "sha256": sha256_file(ARTIFACT_DIR / f"{name}_rf.pkl"),
            },
        },
        "_vocab": vocab,
    }


def write_result_tables(metrics: dict[str, Any]) -> None:
    rows = []
    for name in ["leaky", "clean"]:
        exp = metrics[name]
        for split in ["holdout", "dev", "test"]:
            row = {"experiment": exp["experiment"], "split": split}
            row.update({k: exp[split][k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]})
            rows.append(row)
    pd.DataFrame([row for row in rows if row["split"] == "dev"]).to_csv(REPORT_DIR / "controlled_dev_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([row for row in rows if row["split"] == "test"]).to_csv(REPORT_DIR / "controlled_test_results.csv", index=False, encoding="utf-8-sig")


def write_markdown(metrics: dict[str, Any]) -> None:
    leaky = metrics["leaky"]
    clean = metrics["clean"]
    diff = {
        "holdout_macro_f1": leaky["holdout"]["macro_f1"] - clean["holdout"]["macro_f1"],
        "dev_macro_f1": leaky["dev"]["macro_f1"] - clean["dev"]["macro_f1"],
        "test_macro_f1": leaky["test"]["macro_f1"] - clean["test"]["macro_f1"],
    }
    content = f"""# CONTROLLED_EXPERIMENT_REPORT

## Question

在相同v2数据、相同20000条样本、相同16000条RF训练样本、相同4000条内部holdout、相同RF参数和相同评估集下，只改变TF-IDF fit范围，观察提前看到holdout词表与IDF对指标的影响。

## Controlled Design

- sample source: `01-data/v2/process_train_v2.csv`
- sample size: 20000
- sampler: `StratifiedShuffleSplit(random_state=42)`
- split: 16000 train_core / 4000 internal_holdout, `random_state=5`
- RF params: `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`
- leaky TF-IDF fit rows: {leaky["tfidf_fit_rows"]}
- clean TF-IDF fit rows: {clean["tfidf_fit_rows"]}

## Results

| experiment | holdout macro_f1 | dev macro_f1 | test macro_f1 | vocabulary size |
|---|---:|---:|---:|---:|
| controlled_leaky_20k | {leaky["holdout"]["macro_f1"]:.6f} | {leaky["dev"]["macro_f1"]:.6f} | {leaky["test"]["macro_f1"]:.6f} | {leaky["vocabulary_size"]} |
| controlled_clean_20k | {clean["holdout"]["macro_f1"]:.6f} | {clean["dev"]["macro_f1"]:.6f} | {clean["test"]["macro_f1"]:.6f} | {clean["vocabulary_size"]} |

## Deltas: leaky - clean

- holdout macro_f1: {diff["holdout_macro_f1"]:.6f}
- dev macro_f1: {diff["dev_macro_f1"]:.6f}
- test macro_f1: {diff["test_macro_f1"]:.6f}

## Vocabulary

- leaky vocabulary size: {leaky["vocabulary_size"]}
- clean vocabulary size: {clean["vocabulary_size"]}
- vocabulary intersection: {metrics["vocabulary_intersection_size"]}
- only in leaky vocabulary: {metrics["only_leaky_vocabulary_size"]}

## Boundary

This experiment only isolates TF-IDF validation vocabulary leakage inside the same v2 sample. It does not prove that v2 is better than the original data, does not quantify the full score impact of Day 1 cleaning, and does not claim every course metric is invalid.
"""
    (REPORT_DIR / "CONTROLLED_EXPERIMENT_REPORT.md").write_text(content, encoding="utf-8")


def run_controlled() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    class_names = read_class_names()
    stopwords = read_stopwords()
    train_df = load_csv(DATA_V2 / "process_train_v2.csv")
    dev_df = load_csv(DATA_V2 / "process_dev_v2.csv")
    test_df = load_csv(DATA_V2 / "process_test_v2.csv")
    sampled = sample_indices(train_df)
    sample_df = sampled["sample_df"]
    train_pos = sampled["train_pos"]
    holdout_pos = sampled["holdout_pos"]

    (ARTIFACT_DIR / "shared_sample_indices.json").write_text(
        json.dumps(sampled["payload"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log_lines = ["controlled TF-IDF experiment started"]
    leaky = train_one("leaky", sample_df, train_pos, holdout_pos, dev_df, test_df, stopwords, class_names)
    clean = train_one("clean", sample_df, train_pos, holdout_pos, dev_df, test_df, stopwords, class_names)
    if leaky["rf_params"] != clean["rf_params"]:
        raise AssertionError("RF params differ")
    leaky_vocab = leaky.pop("_vocab")
    clean_vocab = clean.pop("_vocab")
    metrics = {
        "sample": sampled["payload"],
        "leaky": leaky,
        "clean": clean,
        "rf_params_identical": True,
        "only_difference": "TF-IDF fit rows: leaky=20000, clean=16000",
        "vocabulary_intersection_size": len(leaky_vocab & clean_vocab),
        "only_leaky_vocabulary_size": len(leaky_vocab - clean_vocab),
        "only_clean_vocabulary_size": len(clean_vocab - leaky_vocab),
    }
    write_result_tables(metrics)
    write_markdown(metrics)
    (REPORT_DIR / "controlled_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log_lines.append("controlled TF-IDF experiment finished")
    log_lines.append(f"leaky test macro_f1={leaky['test']['macro_f1']}")
    log_lines.append(f"clean test macro_f1={clean['test']['macro_f1']}")
    (REPORT_DIR / "controlled_training_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    run_controlled()
    print("controlled TF-IDF experiment finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
