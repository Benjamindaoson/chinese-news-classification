"""Train TextCNN or BiLSTM baselines on the prepared Chinanews split.

The models share the same tokenizer, validation split, and frozen official test split
as the BERT experiment so that comparisons are meaningful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, 256, kernel_size=k) for k in (2, 3, 4)]
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256 * 3, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids).transpose(1, 2)
        features = [torch.relu(conv(x)).amax(dim=-1) for conv in self.convs]
        x = torch.cat(features, dim=-1)
        return self.classifier(self.dropout(x))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        output, _ = self.lstm(x)
        if attention_mask is None:
            pooled = output.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["textcnn", "bilstm"], required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/chinanews"))
    parser.add_argument("--tokenizer_name", default="bert-base-chinese")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/chinanews-baselines"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            logits = model(**inputs)
            pred = logits.argmax(dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_meta = json.loads((args.data_dir / "labels.json").read_text(encoding="utf-8"))
    labels = label_meta["id_to_label"]
    num_classes = len(labels)

    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(args.data_dir / "train.tsv"),
            "validation": str(args.data_dir / "validation.tsv"),
            "test": str(args.data_dir / "test.tsv"),
        },
        delimiter="\t",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

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
    ).rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    loaders = {
        split: DataLoader(
            tokenized[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
        for split in ("train", "validation", "test")
    }

    if args.model == "textcnn":
        model = TextCNN(tokenizer.vocab_size, num_classes)
    else:
        model = BiLSTM(tokenizer.vocab_size, num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    run_dir = args.output_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in loaders["train"]:
            labels_tensor = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                logits = model(**inputs)
                loss = loss_fn(logits, labels_tensor)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = labels_tensor.size(0)
            total_loss += loss.item() * batch_size
            seen += batch_size

        val_metrics = evaluate(model, loaders["validation"], device)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            **{f"validation_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, loaders["test"], device)
    payload = {
        "model": args.model,
        "labels": labels,
        "best_validation_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
