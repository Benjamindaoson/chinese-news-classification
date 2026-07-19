from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import jieba
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports" / "random_forest"
ARTIFACTS = ROOT / "artifacts" / "random_forest"
BASELINE_REPORTS = REPORTS / "baseline"
V2_REPORTS = REPORTS / "v2"
BASELINE_ARTIFACTS = ARTIFACTS / "baseline"
V2_ARTIFACTS = ARTIFACTS / "v2"
PROTECTED = [
    DATA / "process_train.csv",
    DATA / "process_dev.csv",
    DATA / "process_test.csv",
    DATA / "model" / "tfidf_model.pkl",
    DATA / "model" / "rf_model.pkl",
]


@dataclass(frozen=True)
class Paths:
    train_csv: Path
    dev_csv: Path
    test_csv: Path
    tfidf: Path
    rf: Path
    report_dir: Path
    artifact_dir: Path
    prefix: str


def ensure_dirs() -> None:
    for path in [BASELINE_REPORTS, V2_REPORTS, BASELINE_ARTIFACTS, V2_ARTIFACTS]:
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_class_names() -> list[str]:
    return [line.strip() for line in (DATA / "class.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_stopwords() -> list[str]:
    return [line.strip() for line in (DATA / "stopwords.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def load_process_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["text", "label", "words", "seq_len"]
    if list(df.columns) != expected:
        raise ValueError(f"{path} columns must be {expected}, got {list(df.columns)}")
    if df[["text", "label", "words", "seq_len"]].isna().any().any():
        raise ValueError(f"{path} contains missing values")
    return df


def parse_txt(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw in enumerate(file, 1):
            line = raw.rstrip("\r\n")
            if "\t" not in line:
                raise ValueError(f"{path}:{line_number} missing tab")
            text, label = line.rsplit("\t", 1)
            rows.append((text, int(label)))
    return rows


def process_rows(rows: list[tuple[str, int]]) -> pd.DataFrame:
    texts = [text for text, _ in rows]
    labels = [label for _, label in rows]
    word_lists = [jieba.lcut(text) for text in texts]
    return pd.DataFrame(
        {
            "text": texts,
            "label": labels,
            "words": [" ".join(words[:20]) for words in word_lists],
            "seq_len": [len(words) for words in word_lists],
        }
    )


def generate_v2_process_csv() -> dict[str, int]:
    v2_dir = DATA / "v2"
    paths = {
        "train": (v2_dir / "train_v2.txt", v2_dir / "process_train_v2.csv"),
        "dev": (v2_dir / "dev_v2.txt", v2_dir / "process_dev_v2.csv"),
        "test": (v2_dir / "test_v2.txt", v2_dir / "process_test_v2.csv"),
    }
    counts: dict[str, int] = {}
    for split, (src, dst) in paths.items():
        rows = parse_txt(src)
        df = process_rows(rows)
        df.to_csv(dst, index=False, encoding="utf-8")
        if len(df) != len(rows):
            raise AssertionError(f"{split} process rows mismatch")
        counts[split] = len(df)
    return counts


def evaluate_classifier(
    model: Any,
    vectorizer: Any,
    texts: pd.Series,
    labels: pd.Series,
    split_name: str,
    class_names: list[str],
) -> tuple[dict[str, Any], list[int], Any]:
    features = vectorizer.transform(texts)
    start = time.perf_counter()
    predictions = model.predict(features)
    elapsed = time.perf_counter() - start
    labels_order = list(range(10))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=labels_order, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )[2]
    report_rows = []
    for idx, label in enumerate(labels_order):
        report_rows.append(
            {
                "label": label,
                "class_name": class_names[label] if label < len(class_names) else str(label),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
    metrics = {
        "split": split_name,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "total_inference_seconds": elapsed,
        "average_inference_ms": elapsed / len(labels) * 1000,
        "samples_per_second": len(labels) / elapsed if elapsed else None,
        "per_class": report_rows,
    }
    return metrics, list(predictions), features


def write_metrics(report_dir: Path, prefix: str, metrics: dict[str, Any], labels: pd.Series, predictions: list[int], class_names: list[str]) -> None:
    (report_dir / f"{prefix}_metrics.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["per_class"]).to_csv(report_dir / f"{prefix}_classification_report.csv", index=False, encoding="utf-8-sig")
    cm = confusion_matrix(labels, predictions, labels=list(range(10)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"{i}:{class_names[i]}" for i in range(10)],
        columns=[f"{i}:{class_names[i]}" for i in range(10)],
    )
    cm_df.to_csv(report_dir / f"{prefix}_confusion_matrix.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{prefix} confusion matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(10), [str(i) for i in range(10)])
    ax.set_yticks(range(10), [str(i) for i in range(10)])
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(report_dir / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_models(vectorizer: Any, model: Any, paths: Paths) -> dict[str, Any]:
    joblib.dump(vectorizer, paths.tfidf)
    joblib.dump(model, paths.rf)
    reloaded_vectorizer = joblib.load(paths.tfidf)
    reloaded_model = joblib.load(paths.rf)
    return {
        "tfidf": file_info(paths.tfidf),
        "rf": file_info(paths.rf),
        "reload_check_type": {
            "tfidf": type(reloaded_vectorizer).__name__,
            "rf": type(reloaded_model).__name__,
        },
    }


def smoke_train(df_train: pd.DataFrame, df_eval: pd.DataFrame, stopwords: list[str]) -> None:
    train = df_train.head(1000)
    eval_df = df_eval.head(200)
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    x_train = vectorizer.fit_transform(train["words"])
    model = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=-1)
    model.fit(x_train, train["label"])
    preds = model.predict(vectorizer.transform(eval_df["words"]))
    if len(preds) != len(eval_df):
        raise AssertionError("smoke prediction length mismatch")


def evaluate_existing_course_artifacts() -> dict[str, Any]:
    ensure_dirs()
    class_names = read_class_names()
    test_df = load_process_csv(DATA / "process_test.csv")
    with (DATA / "model" / "rf_model.pkl").open("rb") as file:
        model = pickle.load(file)
    with (DATA / "model" / "tfidf_model.pkl").open("rb") as file:
        vectorizer = pickle.load(file)
    metrics, preds, _ = evaluate_classifier(model, vectorizer, test_df["words"], test_df["label"], "process_test.csv", class_names)
    metrics["experiment"] = "existing_course_artifact"
    metrics["model_files"] = {
        "tfidf": file_info(DATA / "model" / "tfidf_model.pkl"),
        "rf": file_info(DATA / "model" / "rf_model.pkl"),
    }
    write_metrics(BASELINE_REPORTS, "existing_course_artifact", metrics, test_df["label"], preds, class_names)
    return metrics


def train_baseline_reproduction() -> dict[str, Any]:
    ensure_dirs()
    class_names = read_class_names()
    stopwords = read_stopwords()
    train_df_full = load_process_csv(DATA / "process_train.csv")
    test_df = load_process_csv(DATA / "process_test.csv")
    train_df = train_df_full.head(20000)
    smoke_train(train_df, test_df, stopwords)
    start = time.perf_counter()
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    features = vectorizer.fit_transform(train_df["words"])
    x_train, _, y_train, _ = train_test_split(features, train_df["label"], test_size=0.2, random_state=5)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start
    metrics, preds, _ = evaluate_classifier(model, vectorizer, test_df["words"], test_df["label"], "process_test.csv", class_names)
    metrics.update(
        {
            "experiment": "baseline_reproduction",
            "training_seconds": train_seconds,
            "training_rows_read": len(train_df),
            "training_rows_after_split": len(y_train),
            "tfidf_vocabulary_size": len(vectorizer.vocabulary_),
            "validation_vocabulary_leakage": True,
            "course_dev_used_for_training_selection": False,
            "random_forest_params": model.get_params(),
            "tfidf_params": vectorizer.get_params(),
        }
    )
    paths = Paths(
        DATA / "process_train.csv",
        DATA / "process_dev.csv",
        DATA / "process_test.csv",
        BASELINE_ARTIFACTS / "tfidf_baseline.pkl",
        BASELINE_ARTIFACTS / "rf_baseline.pkl",
        BASELINE_REPORTS,
        BASELINE_ARTIFACTS,
        "baseline",
    )
    metrics["saved_models"] = save_models(vectorizer, model, paths)
    write_metrics(BASELINE_REPORTS, "baseline", metrics, test_df["label"], preds, class_names)
    return metrics


def candidate_rf_params() -> list[dict[str, Any]]:
    return [
        {
            "n_estimators": 50,
            "max_depth": 40,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
    ]


def train_v2() -> dict[str, Any]:
    ensure_dirs()
    process_counts = generate_v2_process_csv()
    class_names = read_class_names()
    stopwords = read_stopwords()
    train_df = load_process_csv(DATA / "v2" / "process_train_v2.csv")
    dev_df = load_process_csv(DATA / "v2" / "process_dev_v2.csv")
    test_df = load_process_csv(DATA / "v2" / "process_test_v2.csv")
    smoke_train(train_df, dev_df, stopwords)

    vectorizer = TfidfVectorizer(stop_words=stopwords)
    x_train = vectorizer.fit_transform(train_df["words"])
    x_dev = vectorizer.transform(dev_df["words"])
    best: tuple[float, dict[str, Any], RandomForestClassifier, float] | None = None
    candidates: list[dict[str, Any]] = []

    for params in candidate_rf_params():
        start = time.perf_counter()
        model = RandomForestClassifier(**params)
        model.fit(x_train, train_df["label"])
        train_seconds = time.perf_counter() - start
        dev_pred = model.predict(x_dev)
        macro_f1 = precision_recall_fscore_support(dev_df["label"], dev_pred, average="macro", zero_division=0)[2]
        row = {"params": params, "dev_macro_f1": float(macro_f1), "training_seconds": train_seconds}
        candidates.append(row)
        if best is None or macro_f1 > best[0]:
            best = (float(macro_f1), params, model, train_seconds)

    if best is None:
        raise AssertionError("no v2 candidate trained")
    _, selected_params, model, train_seconds = best
    metrics, preds, _ = evaluate_classifier(model, vectorizer, test_df["words"], test_df["label"], "process_test_v2.csv", class_names)
    metrics.update(
        {
            "experiment": "v2_upgraded",
            "training_seconds": train_seconds,
            "process_counts": process_counts,
            "tfidf_vocabulary_size": len(vectorizer.vocabulary_),
            "selected_params": selected_params,
            "candidate_results": candidates,
            "tfidf_fit_split": "train_v2_only",
            "dev_transform_only": True,
            "test_transform_only": True,
            "test_used_for_model_selection": False,
            "random_forest_params": model.get_params(),
            "tfidf_params": vectorizer.get_params(),
        }
    )
    paths = Paths(
        DATA / "v2" / "process_train_v2.csv",
        DATA / "v2" / "process_dev_v2.csv",
        DATA / "v2" / "process_test_v2.csv",
        V2_ARTIFACTS / "tfidf_v2.pkl",
        V2_ARTIFACTS / "rf_v2.pkl",
        V2_REPORTS,
        V2_ARTIFACTS,
        "v2",
    )
    metrics["saved_models"] = save_models(vectorizer, model, paths)
    write_metrics(V2_REPORTS, "v2", metrics, test_df["label"], preds, class_names)
    write_error_samples(V2_REPORTS / "error_samples.csv", test_df, preds, model, vectorizer, class_names)
    return metrics


def write_error_samples(path: Path, test_df: pd.DataFrame, preds: list[int], model: Any, vectorizer: Any, class_names: list[str]) -> None:
    errors = test_df.copy()
    errors["predicted_label"] = preds
    errors = errors[errors["label"] != errors["predicted_label"]].copy().head(200)
    if hasattr(model, "predict_proba") and len(errors):
        probs = model.predict_proba(vectorizer.transform(errors["words"]))
    else:
        probs = []
    rows: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(errors.iterrows()):
        true_label = int(row["label"])
        pred_label = int(row["predicted_label"])
        predicted_probability = ""
        top2_label = ""
        top2_probability = ""
        if len(errors) and hasattr(model, "predict_proba"):
            pairs = sorted(enumerate(probs[idx]), key=lambda item: item[1], reverse=True)
            predicted_probability = float(dict(pairs).get(pred_label, 0.0))
            if len(pairs) > 1:
                top2_label = int(pairs[1][0])
                top2_probability = float(pairs[1][1])
        rows.append(
            {
                "text": row["text"],
                "true_label": true_label,
                "true_class_name": class_names[true_label],
                "predicted_label": pred_label,
                "predicted_class_name": class_names[pred_label],
                "text_length": len(str(row["text"])),
                "predicted_probability": predicted_probability,
                "top2_label": top2_label,
                "top2_probability": top2_probability,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_text_reports(existing: dict[str, Any] | None, baseline: dict[str, Any] | None, v2: dict[str, Any] | None) -> None:
    if existing or baseline:
        content = f"""# BASELINE REPORT

## Source Code Findings

- `random_forest/data_process.py` reads `train.txt`, `test.txt`, and `dev.txt`, applies `jieba.lcut(text)[:20]` into `words`, computes `seq_len` as full jieba token count, and writes `process_*.csv`.
- It imports stopwords only in training, not preprocessing. Stopwords are passed to `TfidfVectorizer(stop_words=stopwords)`.
- `process_*.csv` columns are `text,label,words,seq_len`.
- `random_forest/rf_train.py` reads `process_train.csv` and truncates to the first 20000 rows.
- It calls `TfidfVectorizer(...).fit_transform(words)` before `train_test_split(..., test_size=0.2, random_state=5)`.
- It does not use the course `process_dev.csv` during training or model selection.
- Original RF params are `RandomForestClassifier(n_estimators=100)` with default sklearn values.
- Original files save to `data/model/rf_model.pkl` and `data/model/tfidf_model.pkl`; this reproduction writes only to `artifacts/random_forest/baseline`.

## Leakage

`validation_vocabulary_leakage = true` for the course training logic, because TF-IDF sees the entire 20000-row training sample before the internal validation split.

## Metrics

| experiment | accuracy | macro_f1 | weighted_f1 |
|---|---:|---:|---:|
| existing_course_artifact | {existing.get('accuracy') if existing else ''} | {existing.get('macro_f1') if existing else ''} | {existing.get('weighted_f1') if existing else ''} |
| baseline_reproduction | {baseline.get('accuracy') if baseline else ''} | {baseline.get('macro_f1') if baseline else ''} | {baseline.get('weighted_f1') if baseline else ''} |

## Difference From Course Code

The reproduction keeps the course TF-IDF order and 20000-row truncation, but adds `random_state=42` and `n_jobs=-1` to RandomForest for reproducibility and runtime. It does not overwrite original course artifacts.
"""
        (BASELINE_REPORTS / "BASELINE_REPORT.md").write_text(content, encoding="utf-8")

    if v2:
        worst = sorted(v2["per_class"], key=lambda row: row["f1"])[:3]
        content = f"""# V2 REPORT

## Data Source

v2 uses `data/v2/train_v2.txt`, `dev_v2.txt`, and `test_v2.txt`. Generated process CSV files are under `data/v2/process_*_v2.csv`.

## Strict TF-IDF Flow

TF-IDF is fit only on train_v2. dev_v2 and test_v2 use transform only. test_v2 is not used for parameter selection.

## Selected Parameters

`{v2['selected_params']}`

## Results

- test accuracy: {v2['accuracy']}
- test macro_f1: {v2['macro_f1']}
- test weighted_f1: {v2['weighted_f1']}
- vocabulary size: {v2['tfidf_vocabulary_size']}
- training seconds: {v2['training_seconds']}
- average inference ms: {v2['average_inference_ms']}

## Worst Classes

{chr(10).join(f"- {row['label']}:{row['class_name']} f1={row['f1']}" for row in worst)}

## Error Samples

See `reports/random_forest/v2/error_samples.csv`. RandomForest on short Chinese titles mainly fails around short titles, mixed-topic titles, fuzzy class boundaries, proper nouns, numbers/abbreviations, segmentation limitations, stopword effects, and label noise. It should not be described as understanding news semantics.

## Saved Model Location

- `artifacts/random_forest/v2/tfidf_v2.pkl`
- `artifacts/random_forest/v2/rf_v2.pkl`

## Reproducible Command

`python random_forest/run_random_forest.py --mode v2`
"""
        (V2_REPORTS / "V2_REPORT.md").write_text(content, encoding="utf-8")

    if existing and baseline and v2:
        rows = [
            ("existing_course_artifact", "old process_test.csv", "yes", "yes", existing),
            ("baseline_reproduction", "old process_train/test.csv", "yes", "yes", baseline),
            ("v2_upgraded", "v2 txt/csv", "no", "no", v2),
        ]
        content = """# RANDOM FOREST COMPARISON

| experiment | data | cross-split duplicate | TF-IDF leakage | Accuracy | Macro-F1 | Weighted-F1 | model size | training seconds |
|---|---|---|---|---:|---:|---:|---:|---:|
"""
        for name, data, dup, leak, m in rows:
            model_size = m.get("saved_models", m.get("model_files", {})).get("rf", {}).get("size_bytes", "")
            content += f"| {name} | {data} | {dup} | {leak} | {m['accuracy']} | {m['macro_f1']} | {m['weighted_f1']} | {model_size} | {m.get('training_seconds', '')} |\n"
        content += """

v2 scores may be lower than course baseline because cross-split leakage and TF-IDF validation leakage were removed. A lower score after leakage repair does not mean the model became worse; it means the estimate is more trustworthy. RandomForest is a useful baseline for sparse text features, but short Chinese titles usually need FastText/BERT follow-up for stronger representation.
"""
        (REPORTS / "RANDOM_FOREST_COMPARISON.md").write_text(content, encoding="utf-8")


def protected_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in PROTECTED if path.exists()}


def run(mode: str) -> None:
    ensure_dirs()
    if mode == "controlled":
        from controlled.controlled_tfidf_experiment import run_controlled

        run_controlled()
        return
    before = protected_hashes()
    existing = baseline = v2 = None
    if mode in {"evaluate-existing", "all"}:
        existing = evaluate_existing_course_artifacts()
    if mode in {"baseline", "all"}:
        log_path = BASELINE_REPORTS / "baseline_training_log.txt"
        with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
            baseline = train_baseline_reproduction()
    if mode in {"v2", "all"}:
        log_path = V2_REPORTS / "v2_training_log.txt"
        with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
            v2 = train_v2()
    if mode != "all":
        if (BASELINE_REPORTS / "existing_course_artifact_metrics.json").exists():
            existing = json.loads((BASELINE_REPORTS / "existing_course_artifact_metrics.json").read_text(encoding="utf-8"))
        if (BASELINE_REPORTS / "baseline_metrics.json").exists():
            baseline = json.loads((BASELINE_REPORTS / "baseline_metrics.json").read_text(encoding="utf-8"))
        if (V2_REPORTS / "v2_metrics.json").exists():
            v2 = json.loads((V2_REPORTS / "v2_metrics.json").read_text(encoding="utf-8"))
    write_text_reports(existing, baseline, v2)
    after = protected_hashes()
    verification = {
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_hashes_unchanged": before == after,
        "mode": mode,
    }
    (REPORTS / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    if before != after:
        raise AssertionError("protected original files changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "v2", "all", "evaluate-existing", "controlled"], required=True)
    args = parser.parse_args()
    run(args.mode)
    print(f"random_forest mode {args.mode} finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
