import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class Metrics:
    acc: float
    f1: float
    precision: float
    recall: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return Metrics(acc=acc, f1=float(f1), precision=float(precision), recall=float(recall))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_xy(csv_path: Path, text_col: str, label_col: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if text_col not in df.columns:
        raise ValueError(f"Missing text_col '{text_col}' in {csv_path}. cols={list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label_col '{label_col}' in {csv_path}. cols={list(df.columns)}")

    x = (
        df[text_col]
        .fillna("")
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .tolist()
    )

    y = pd.to_numeric(df[label_col], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Found NaN labels in {csv_path}")
    y = y.astype(int).to_numpy()

    bad = set(np.unique(y)) - {0, 1}
    if bad:
        raise ValueError(f"Label must be 0/1, found {sorted(bad)} in {csv_path}")

    return x, y


class RobertaDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_len: int) -> None:
        self.texts = texts
        self.labels = labels.astype(np.int64)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _default_device(user_device: str | None) -> str:
    if user_device:
        return user_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None,
) -> float:
    model.train()
    losses: list[float] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def eval_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[Metrics, dict]:
    model.eval()

    all_probs: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []

    t0 = time.time()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]

        all_probs.append(probs.detach().cpu())
        all_y.append(labels.detach().cpu())

    dt = time.time() - t0

    probs = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_y, dim=0).numpy().astype(int)
    y_pred = (probs >= threshold).astype(int)

    m = compute_metrics(y_true, y_pred)
    extra = {
        "examples": int(len(y_true)),
        "seconds": float(dt),
        "examples_per_second": float(len(y_true) / dt) if dt > 0 else None,
    }
    return m, extra


def main() -> int:
    parser = argparse.ArgumentParser(description="RoBERTa baseline for emoji irony detection.")
    parser.add_argument("--train_csv", type=str, default="data/emoji_train.csv")
    parser.add_argument("--val_csv", type=str, default="data/emoji_val.csv")
    parser.add_argument("--test_csv", type=str, default="data/emoji_test.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="irony")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        type=str,
        default="both",
        choices=["val", "test", "both"],
        help="Which split(s) to evaluate.",
    )
    parser.add_argument("--output_dir", type=str, default="artifacts/baselines/roberta")

    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    train_path = Path(args.train_csv)
    val_path = Path(args.val_csv)
    test_path = Path(args.test_csv)

    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(p)

    X_train, y_train = read_xy(train_path, args.text_col, args.label_col)
    X_val, y_val = read_xy(val_path, args.text_col, args.label_col)
    X_test, y_test = read_xy(test_path, args.text_col, args.label_col)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        use_safetensors=False,
    )

    ds_train = RobertaDataset(X_train, y_train, tokenizer, args.max_len)
    ds_val = RobertaDataset(X_val, y_val, tokenizer, args.max_len)
    ds_test = RobertaDataset(X_test, y_test, tokenizer, args.max_len)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    device = torch.device(_default_device(args.device))
    model.to(device)

    train_pos = float(np.mean(y_train == 1))
    train_neg = 1.0 - train_pos
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    class_weights = torch.tensor([1.0, neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    t_train0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=dl_train,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_m, _ = eval_split(model, dl_val, device=device, threshold=args.threshold)

        print(
            f"epoch={epoch} loss={train_loss:.4f} "
            f"val_f1={val_m.f1:.4f} val_acc={val_m.acc:.4f}"
        )

        if val_m.f1 > best_f1 + 1e-6:
            best_f1 = float(val_m.f1)
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            break

    train_seconds = float(time.time() - t_train0)

    if best_state is not None:
        model.load_state_dict(best_state)

    results: dict[str, dict] = {
        "meta": {
            "provider": "huggingface",
            "method": "roberta_finetune",
            "model": args.model_name,
            "model_name": args.model_name,
            "splits": args.splits,
            "seed": args.seed,
            "threshold": args.threshold,
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "best_epoch": int(best_epoch),
            "train_seconds": train_seconds,
            "class_weights": class_weights.detach().cpu().tolist(),
            "device": str(device),
            "splits": {
                "train": str(train_path),
                "val": str(val_path),
                "test": str(test_path),
            },
        },
    }

    if args.splits in ("val", "both"):
        m, extra = eval_split(model, dl_val, device=device, threshold=args.threshold)
        results["val"] = {**m.__dict__, **extra}

    if args.splits in ("test", "both"):
        m, extra = eval_split(model, dl_test, device=device, threshold=args.threshold)
        results["test"] = {**m.__dict__, **extra}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = out_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("Saved to:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
