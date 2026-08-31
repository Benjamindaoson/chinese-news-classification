"""Fine-tune bert-base-chinese on the prepared Chinanews split.

The official test split is evaluated only after validation-based checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/chinanews"))
    parser.add_argument("--model_name", default="bert-base-chinese")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/chinanews-bert"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_meta = json.loads((args.data_dir / "labels.json").read_text(encoding="utf-8"))
    labels = label_meta["id_to_label"]
    num_labels = len(labels)
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in id2label.items()}

    data_files = {
        "train": str(args.data_dir / "train.tsv"),
        "validation": str(args.data_dir / "validation.tsv"),
        "test": str(args.data_dir / "test.tsv"),
    }
    dataset = load_dataset(
        "csv",
        data_files=data_files,
        delimiter="\t",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing Chinanews",
    )
    tokenized = tokenized.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.num_workers,
        group_by_length=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )

    train_result = trainer.train()
    trainer.save_model(str(args.output_dir / "best_model"))
    tokenizer.save_pretrained(str(args.output_dir / "best_model"))

    # The official test split is touched only after validation-based model selection.
    test_output = trainer.predict(tokenized["test"], metric_key_prefix="test")
    test_pred = np.argmax(test_output.predictions, axis=-1)
    test_true = test_output.label_ids

    report = classification_report(
        test_true,
        test_pred,
        labels=list(range(num_labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_true, test_pred, labels=list(range(num_labels)))

    payload = {
        "model_name": args.model_name,
        "seed": args.seed,
        "train_metrics": train_result.metrics,
        "test_metrics": test_output.metrics,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
    }
    (args.output_dir / "final_test_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trainer.save_state()

    print(json.dumps(test_output.metrics, ensure_ascii=False, indent=2))
    print(f"Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
