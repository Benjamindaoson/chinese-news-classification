from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


VALID_LABELS = set(range(10))
INPUT_SPLITS = ("train", "dev", "test")
PRIORITY = ("test", "dev", "train")
REMOVED_FIELDS = [
    "source_split",
    "original_line_number",
    "original_text",
    "normalized_text",
    "label",
    "removal_reason",
    "duplicate_of_split",
]


@dataclass(frozen=True)
class Record:
    source_split: str
    line_number: int
    text: str
    normalized_text: str
    label: str


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def removed_row(
    split: str,
    line_number: int,
    text: str,
    normalized_text: str,
    label: str,
    reason: str,
    duplicate_of_split: str = "",
) -> dict[str, str | int]:
    return {
        "source_split": split,
        "original_line_number": line_number,
        "original_text": text,
        "normalized_text": normalized_text,
        "label": label,
        "removal_reason": reason,
        "duplicate_of_split": duplicate_of_split,
    }


def parse_dataset(path: Path, split: str) -> tuple[list[Record], list[dict[str, str | int]], Counter[str]]:
    records: list[Record] = []
    removed: list[dict[str, str | int]] = []
    issue_counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                issue_counts["empty_line"] += 1
                removed.append(removed_row(split, line_number, "", "", "", "empty_line"))
                continue

            if "\t" not in line:
                issue_counts["missing_tab"] += 1
                removed.append(
                    removed_row(split, line_number, line, normalize_text(line), "", "missing_tab")
                )
                continue

            text, label = line.rsplit("\t", 1)
            normalized = normalize_text(text)

            if not normalized:
                issue_counts["empty_text"] += 1
                removed.append(removed_row(split, line_number, text, normalized, label, "empty_text"))
                continue
            if label == "":
                issue_counts["empty_label"] += 1
                removed.append(removed_row(split, line_number, text, normalized, label, "empty_label"))
                continue
            try:
                label_int = int(label)
            except ValueError:
                issue_counts["invalid_label"] += 1
                removed.append(removed_row(split, line_number, text, normalized, label, "invalid_label"))
                continue
            if label_int not in VALID_LABELS:
                issue_counts["label_out_of_range"] += 1
                removed.append(
                    removed_row(split, line_number, text, normalized, label, "label_out_of_range")
                )
                continue

            records.append(Record(split, line_number, text, normalized, label))

    return records, removed, issue_counts


def find_conflicting_labels(records_by_split: dict[str, list[Record]]) -> set[str]:
    labels_by_text: dict[str, set[str]] = defaultdict(set)
    for records in records_by_split.values():
        for record in records:
            labels_by_text[record.normalized_text].add(record.label)
    return {text for text, labels in labels_by_text.items() if len(labels) > 1}


def deduplicate_by_priority(
    records_by_split: dict[str, list[Record]],
    conflict_keys: set[str],
) -> tuple[dict[str, list[Record]], list[dict[str, str | int]]]:
    kept: dict[str, list[Record]] = {split: [] for split in INPUT_SPLITS}
    removed: list[dict[str, str | int]] = []

    for split in INPUT_SPLITS:
        for record in records_by_split[split]:
            if record.normalized_text in conflict_keys:
                removed.append(
                    removed_row(
                        record.source_split,
                        record.line_number,
                        record.text,
                        record.normalized_text,
                        record.label,
                        "conflicting_labels",
                    )
                )

    owner_by_text: dict[str, str] = {}
    for split in PRIORITY:
        seen_in_split: set[str] = set()
        for record in records_by_split[split]:
            if record.normalized_text in conflict_keys:
                continue

            owner = owner_by_text.get(record.normalized_text)
            if owner:
                reason = "duplicate_with_test" if owner == "test" else "duplicate_with_dev"
                removed.append(
                    removed_row(
                        record.source_split,
                        record.line_number,
                        record.text,
                        record.normalized_text,
                        record.label,
                        reason,
                        owner,
                    )
                )
                continue

            if record.normalized_text in seen_in_split:
                removed.append(
                    removed_row(
                        record.source_split,
                        record.line_number,
                        record.text,
                        record.normalized_text,
                        record.label,
                        "duplicate_within_split",
                        split,
                    )
                )
                continue

            seen_in_split.add(record.normalized_text)
            owner_by_text[record.normalized_text] = split
            kept[split].append(record)

    return kept, removed


def write_dataset(path: Path, records: Iterable[Record]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(f"{record.text}\t{record.label}\n")


def write_removed_samples(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REMOVED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def label_distribution(records: Iterable[Record]) -> dict[str, int]:
    return dict(sorted(Counter(record.label for record in records).items(), key=lambda item: int(item[0])))


def text_length_stats(records: Iterable[Record]) -> dict[str, float | int | None]:
    lengths = [len(record.text) for record in records]
    if not lengths:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(lengths),
        "mean": sum(lengths) / len(lengths),
        "min": min(lengths),
        "max": max(lengths),
    }


def normalized_keys(records: Iterable[Record]) -> set[str]:
    return {record.normalized_text for record in records}


def validate_v2(
    kept: dict[str, list[Record]],
    input_hash_before: dict[str, str],
    input_hash_after: dict[str, str],
) -> dict[str, object]:
    results: dict[str, object] = {}
    key_sets = {split: normalized_keys(records) for split, records in kept.items()}

    for split, records in kept.items():
        keys = [record.normalized_text for record in records]
        labels = {record.label for record in records}
        results[f"{split}_internal_duplicate_count"] = len(keys) - len(set(keys))
        results[f"{split}_non_empty"] = len(records) > 0
        results[f"{split}_has_10_classes"] = labels == {str(i) for i in range(10)}
        results[f"{split}_all_rows_valid"] = all(
            record.text and record.label.isdigit() and int(record.label) in VALID_LABELS
            for record in records
        )

    results["train_dev_intersection"] = len(key_sets["train"] & key_sets["dev"])
    results["train_test_intersection"] = len(key_sets["train"] & key_sets["test"])
    results["dev_test_intersection"] = len(key_sets["dev"] & key_sets["test"])
    results["input_hashes_unchanged"] = input_hash_before == input_hash_after

    expected_zero = [
        "train_internal_duplicate_count",
        "dev_internal_duplicate_count",
        "test_internal_duplicate_count",
        "train_dev_intersection",
        "train_test_intersection",
        "dev_test_intersection",
    ]
    ok = all(results[name] == 0 for name in expected_zero)
    ok = ok and all(results[f"{split}_non_empty"] for split in INPUT_SPLITS)
    ok = ok and all(results[f"{split}_has_10_classes"] for split in INPUT_SPLITS)
    ok = ok and all(results[f"{split}_all_rows_valid"] for split in INPUT_SPLITS)
    ok = ok and bool(results["input_hashes_unchanged"])
    results["all_checks_passed"] = ok
    return results


def build_report(
    data_dir: Path,
    v2_dir: Path,
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    input_hash_before: dict[str, str],
    input_hash_after: dict[str, str],
    records_by_split: dict[str, list[Record]],
    kept: dict[str, list[Record]],
    format_issue_counts: dict[str, Counter[str]],
    removed_rows: list[dict[str, str | int]],
    conflict_keys: set[str],
    validation: dict[str, object],
) -> dict[str, object]:
    reason_counts = Counter(str(row["removal_reason"]) for row in removed_rows)
    output_hashes = {split: calculate_sha256(path) for split, path in output_paths.items()}

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "input_files": {split: str(path.resolve()) for split, path in input_paths.items()},
        "output_files": {split: str(path.resolve()) for split, path in output_paths.items()},
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "input_hashes_unchanged": input_hash_before == input_hash_after,
        "output_sha256": output_hashes,
        "original_line_counts": {split: len(records_by_split[split]) for split in INPUT_SPLITS},
        "v2_line_counts": {split: len(kept[split]) for split in INPUT_SPLITS},
        "format_issue_counts": {
            split: dict(format_issue_counts[split]) for split in INPUT_SPLITS
        },
        "internal_duplicate_counts": {
            split: sum(
                1
                for row in removed_rows
                if row["source_split"] == split and row["removal_reason"] == "duplicate_within_split"
            )
            for split in INPUT_SPLITS
        },
        "cross_split_duplicate_counts": {
            "duplicate_with_test": reason_counts.get("duplicate_with_test", 0),
            "duplicate_with_dev": reason_counts.get("duplicate_with_dev", 0),
        },
        "conflicting_text_count": len(conflict_keys),
        "conflicting_row_count": reason_counts.get("conflicting_labels", 0),
        "removed_samples_total": len(removed_rows),
        "removal_reason_counts": dict(sorted(reason_counts.items())),
        "label_distribution_original": {
            split: label_distribution(records_by_split[split]) for split in INPUT_SPLITS
        },
        "label_distribution_v2": {split: label_distribution(kept[split]) for split in INPUT_SPLITS},
        "text_length_stats_original": {
            split: text_length_stats(records_by_split[split]) for split in INPUT_SPLITS
        },
        "text_length_stats_v2": {split: text_length_stats(kept[split]) for split in INPUT_SPLITS},
        "normalization_rules": [
            'unicodedata.normalize("NFKC", text)',
            "strip leading and trailing whitespace",
            "collapse consecutive whitespace to one ASCII space",
            "normalization is used only as the duplicate-detection key",
        ],
        "priority_rule": "test > dev > train",
        "intersection_validation": {
            "train_dev": validation["train_dev_intersection"],
            "train_test": validation["train_test_intersection"],
            "dev_test": validation["dev_test_intersection"],
        },
        "validation": validation,
        "data_dir": str(data_dir.resolve()),
        "v2_dir": str(v2_dir.resolve()),
    }


def write_json_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in counts.items())


def write_dataset_markdown(path: Path, report: dict[str, object]) -> None:
    original_counts = report["original_line_counts"]
    v2_counts = report["v2_line_counts"]
    reason_counts = report["removal_reason_counts"]
    validation = report["validation"]

    content = f"""# 1. 为什么创建v2

原课程数据可以复现课程基线，但 `train.txt`、`dev.txt`、`test.txt` 之间存在跨集合重复样本。v2数据集用于后续更严格的实验，目标是保留课程原始数据不变，同时生成一套无跨集合泄漏的数据。

# 2. 原数据存在的问题

- cleaned数据内部已经去重，但跨集合仍可能出现相同或空白差异导致的重复文本。
- 同一规范化文本如果对应多个标签，会造成标签冲突。
- 原始清洗过程缺少独立的可审计清洗报告。

# 3. 原始数据规模

| split | 行数 |
|---|---:|
| train | {original_counts["train"]} |
| dev | {original_counts["dev"]} |
| test | {original_counts["test"]} |

# 4. 清洗和去重规则

- 输入只使用 `train.txt`、`dev.txt`、`test.txt`。
- 每行使用 `rsplit("\\t", 1)` 解析为文本和标签。
- 规范化文本只用于判断重复，不覆盖输出正文。
- 规范化步骤：NFKC、去首尾空白、连续空白压缩为一个普通空格。
- 不删除标点、数字、英文，不转换繁简体，不做中文分词，不改写原始标题。

# 5. 为什么使用test > dev > train

test是最终评估集，dev是调参验证集，train是学习材料。发现同一文本同时出现在学习材料和评估材料中时，应优先保留评估材料，删除低优先级集合中的重复样本。因此采用固定优先级 `test > dev > train`，不重新随机划分，不移动样本。

# 6. 标签冲突如何处理

同一规范化文本如果对应多个不同标签，所有相关记录都从 train/dev/test 中移除，原因记录为 `conflicting_labels`。本次 `conflicting_text_count` 为 {report["conflicting_text_count"]}，`conflicting_row_count` 为 {report["conflicting_row_count"]}。

# 7. 原数据与v2数量对比

| split | 原始行数 | v2行数 | 删除数 |
|---|---:|---:|---:|
| train | {original_counts["train"]} | {v2_counts["train"]} | {original_counts["train"] - v2_counts["train"]} |
| dev | {original_counts["dev"]} | {v2_counts["dev"]} | {original_counts["dev"] - v2_counts["dev"]} |
| test | {original_counts["test"]} | {v2_counts["test"]} | {original_counts["test"] - v2_counts["test"]} |

# 8. 被删除样本统计

| removal_reason | 数量 |
|---|---:|
"""
    for reason, count in reason_counts.items():
        content += f"| {reason} | {count} |\n"

    content += f"""
# 9. 原始与v2标签分布

原始标签分布见 `data_cleaning_report.json` 的 `label_distribution_original`。

v2标签分布：

| split | 分布 |
|---|---|
| train | {format_counts(report["label_distribution_v2"]["train"])} |
| dev | {format_counts(report["label_distribution_v2"]["dev"])} |
| test | {format_counts(report["label_distribution_v2"]["test"])} |

# 10. 文本长度统计

详细统计见 `data_cleaning_report.json` 的 `text_length_stats_original` 和 `text_length_stats_v2`。

# 11. 完整性验证

- train_v2内部规范化文本重复数：{validation["train_internal_duplicate_count"]}
- dev_v2内部规范化文本重复数：{validation["dev_internal_duplicate_count"]}
- test_v2内部规范化文本重复数：{validation["test_internal_duplicate_count"]}
- train_v2 与 dev_v2 交集：{validation["train_dev_intersection"]}
- train_v2 与 test_v2 交集：{validation["train_test_intersection"]}
- dev_v2 与 test_v2 交集：{validation["dev_test_intersection"]}
- 三个v2集合均非空：{validation["train_non_empty"] and validation["dev_non_empty"] and validation["test_non_empty"]}
- 每个v2集合仍包含10个类别：{validation["train_has_10_classes"] and validation["dev_has_10_classes"] and validation["test_has_10_classes"]}

# 12. 原始文件哈希验证

原始课程数据没有被修改。运行前后SHA-256是否一致：{report["input_hashes_unchanged"]}。

输入文件运行前SHA-256：

- train: `{report["input_sha256_before"]["train"]}`
- dev: `{report["input_sha256_before"]["dev"]}`
- test: `{report["input_sha256_before"]["test"]}`

输入文件运行后SHA-256：

- train: `{report["input_sha256_after"]["train"]}`
- dev: `{report["input_sha256_after"]["dev"]}`
- test: `{report["input_sha256_after"]["test"]}`

# 13. 后续模型如何使用v2

后续Day 2升级实验应读取：

- `01-data/v2/train_v2.txt`
- `01-data/v2/dev_v2.txt`
- `01-data/v2/test_v2.txt`

现有 `process_*.csv`、`word_*.txt`、`char_*.txt` 仍对应旧数据。进入Day 2时应从v2重新生成升级版中间数据。

# 14. 已知限制

- 当前没有引入任何外部数据。
- 当前没有重新随机划分数据。
- v2只解决格式异常、标签冲突、内部重复和跨集合重复问题。
- 原数据仍用于复现课程基线，v2用于正式升级实验。
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    data_dir = Path(__file__).resolve().parent
    v2_dir = data_dir / "v2"
    v2_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {split: data_dir / f"{split}.txt" for split in INPUT_SPLITS}
    for split, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"缺少输入文件: {split} -> {path}")

    input_hash_before = {split: calculate_sha256(path) for split, path in input_paths.items()}

    records_by_split: dict[str, list[Record]] = {}
    format_removed: list[dict[str, str | int]] = []
    format_issue_counts: dict[str, Counter[str]] = {}
    for split in INPUT_SPLITS:
        records, removed, issue_counts = parse_dataset(input_paths[split], split)
        records_by_split[split] = records
        format_removed.extend(removed)
        format_issue_counts[split] = issue_counts

    conflict_keys = find_conflicting_labels(records_by_split)
    kept, dedup_removed = deduplicate_by_priority(records_by_split, conflict_keys)
    removed_rows = format_removed + dedup_removed

    output_paths = {
        "train": v2_dir / "train_v2.txt",
        "dev": v2_dir / "dev_v2.txt",
        "test": v2_dir / "test_v2.txt",
    }
    for split in INPUT_SPLITS:
        write_dataset(output_paths[split], kept[split])
    write_removed_samples(v2_dir / "removed_samples.csv", removed_rows)

    input_hash_after = {split: calculate_sha256(path) for split, path in input_paths.items()}
    validation = validate_v2(kept, input_hash_before, input_hash_after)
    report = build_report(
        data_dir,
        v2_dir,
        input_paths,
        output_paths,
        input_hash_before,
        input_hash_after,
        records_by_split,
        kept,
        format_issue_counts,
        removed_rows,
        conflict_keys,
        validation,
    )
    write_json_report(v2_dir / "data_cleaning_report.json", report)
    write_dataset_markdown(v2_dir / "DATASET_V2.md", report)

    if not validation["all_checks_passed"]:
        print("v2数据集验证失败，详见 data_cleaning_report.json", file=sys.stderr)
        return 1

    print("v2 dataset build succeeded")
    print(f"train: {len(records_by_split['train'])} -> {len(kept['train'])}")
    print(f"dev: {len(records_by_split['dev'])} -> {len(kept['dev'])}")
    print(f"test: {len(records_by_split['test'])} -> {len(kept['test'])}")
    print(f"removed: {len(removed_rows)}")
    print(f"report: {v2_dir / 'data_cleaning_report.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"构建v2数据集失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
