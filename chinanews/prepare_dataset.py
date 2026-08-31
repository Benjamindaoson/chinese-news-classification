"""Prepare the 1.512M-sample Chinanews dataset for reproducible classification experiments.

Design goals:
- keep the official test split untouched for model selection;
- carve validation data only from the official training split;
- normalize text and remove exact duplicates;
- avoid loading all raw texts into memory;
- write simple TSV files consumable by Hugging Face Datasets and PyTorch.

Expected default CSV layout (no header):
    label,text...
The first column is treated as the label and all remaining columns are joined as text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def label_sort_key(value: str):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def iter_csv_rows(path: Path, delimiter: str, has_header: bool) -> Iterable[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            raw_label = row[0].strip()
            raw_text = " ".join(part for part in row[1:] if part)
            yield raw_label, raw_text


def collect_labels(paths: list[Path], delimiter: str, has_header: bool) -> list[str]:
    labels: set[str] = set()
    for path in paths:
        for label, _ in iter_csv_rows(path, delimiter, has_header):
            if label:
                labels.add(label)
    return sorted(labels, key=label_sort_key)


def write_header(handle) -> None:
    handle.write("text\tlabel\n")


def validation_bucket(key: str, validation_ratio: float) -> bool:
    """Deterministic hash split; approximately stratified over a large balanced dataset."""
    digest = stable_hash(key)
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < validation_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/chinanews"))
    parser.add_argument("--validation_ratio", type=float, default=0.05)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--has_header", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.validation_ratio < 0.5:
        raise ValueError("validation_ratio must be between 0 and 0.5")
    if not args.train_csv.exists() or not args.test_csv.exists():
        raise FileNotFoundError("train_csv and test_csv must exist locally")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = collect_labels(
        [args.train_csv, args.test_csv], args.delimiter, args.has_header
    )
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    (args.output_dir / "labels.json").write_text(
        json.dumps(
            {"label_to_id": label_to_id, "id_to_label": labels},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "source": "Chinanews",
        "validation_ratio": args.validation_ratio,
        "num_classes": len(labels),
        "labels": labels,
        "splits": {},
        "dropped": Counter(),
    }

    test_hashes: set[str] = set()
    test_counts: Counter[int] = Counter()
    test_path = args.output_dir / "test.tsv"

    with test_path.open("w", encoding="utf-8", newline="") as out:
        write_header(out)
        for raw_label, raw_text in iter_csv_rows(
            args.test_csv, args.delimiter, args.has_header
        ):
            text = normalize_text(raw_text)
            if not text or raw_label not in label_to_id:
                report["dropped"]["invalid_test"] += 1
                continue
            text_hash = stable_hash(text)
            if text_hash in test_hashes:
                report["dropped"]["duplicate_within_test"] += 1
                continue
            test_hashes.add(text_hash)
            label = label_to_id[raw_label]
            out.write(f"{text}\t{label}\n")
            test_counts[label] += 1

    train_hashes: set[str] = set()
    train_counts: Counter[int] = Counter()
    val_counts: Counter[int] = Counter()
    train_path = args.output_dir / "train.tsv"
    val_path = args.output_dir / "validation.tsv"

    with train_path.open("w", encoding="utf-8", newline="") as train_out, val_path.open(
        "w", encoding="utf-8", newline=""
    ) as val_out:
        write_header(train_out)
        write_header(val_out)

        for raw_label, raw_text in iter_csv_rows(
            args.train_csv, args.delimiter, args.has_header
        ):
            text = normalize_text(raw_text)
            if not text or raw_label not in label_to_id:
                report["dropped"]["invalid_train"] += 1
                continue

            text_hash = stable_hash(text)
            if text_hash in test_hashes:
                report["dropped"]["train_test_overlap"] += 1
                continue
            if text_hash in train_hashes:
                report["dropped"]["duplicate_within_train"] += 1
                continue
            train_hashes.add(text_hash)

            label = label_to_id[raw_label]
            if validation_bucket(f"{raw_label}:{text_hash}", args.validation_ratio):
                val_out.write(f"{text}\t{label}\n")
                val_counts[label] += 1
            else:
                train_out.write(f"{text}\t{label}\n")
                train_counts[label] += 1

    def split_payload(counter: Counter[int]):
        return {
            "size": int(sum(counter.values())),
            "class_distribution": {
                labels[idx]: int(counter[idx]) for idx in range(len(labels))
            },
        }

    report["splits"] = {
        "train": split_payload(train_counts),
        "validation": split_payload(val_counts),
        "test": split_payload(test_counts),
    }
    report["dropped"] = dict(report["dropped"])

    report_path = args.output_dir / "dataset_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Prepared data written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
