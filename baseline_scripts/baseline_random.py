import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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


def read_xy(csv_path: Path, text_col: str, label_col: str) -> tuple[pd.Series, np.ndarray]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if text_col not in df.columns:
        raise ValueError(f"Missing text_col '{text_col}' in {csv_path}. cols={list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label_col '{label_col}' in {csv_path}. cols={list(df.columns)}")

    x = df[text_col].astype(str)
    y = pd.to_numeric(df[label_col], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Found NaN labels in {csv_path}")
    y = y.astype(int).to_numpy()

    bad = set(np.unique(y)) - {0, 1}
    if bad:
        raise ValueError(f"Label must be 0/1, found {sorted(bad)} in {csv_path}")
    return x, y


def main() -> int:
    parser = argparse.ArgumentParser(description="Lower baselines for emoji irony detection.")
    parser.add_argument("--train_csv", type=str, default="data/emoji_train.csv")
    parser.add_argument("--val_csv", type=str, default="data/emoji_val.csv")
    parser.add_argument("--test_csv", type=str, default="data/emoji_test.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="irony")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        type=str,
        default="both",
        choices=["val", "test", "both"],
        help="Which split(s) to evaluate.",
    )
    parser.add_argument("--output_dir", type=str, default="artifacts/baselines/random")
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    val_path = Path(args.val_csv)
    test_path = Path(args.test_csv)

    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(p)

    _, y_train = read_xy(train_path, args.text_col, args.label_col)
    _, y_val = read_xy(val_path, args.text_col, args.label_col)
    _, y_test = read_xy(test_path, args.text_col, args.label_col)

    train_pos = float(np.mean(y_train == 1))
    train_neg = 1.0 - train_pos

    majority_class = int(np.argmax([train_neg, train_pos]))  # 0 if neg>=pos else 1

    rng = np.random.default_rng(args.seed)

    def majority_pred(y: np.ndarray) -> np.ndarray:
        return np.full_like(y, fill_value=majority_class)

    def stratified_random_pred(y: np.ndarray) -> np.ndarray:
        # Sample according to train label distribution
        return rng.choice([0, 1], size=len(y), replace=True, p=[train_neg, train_pos]).astype(int)

    results: dict[str, dict] = {
        "meta": {
            "seed": args.seed,
            "train_pos_rate": train_pos,
            "train_neg_rate": train_neg,
            "majority_class": majority_class,
            "splits": {
                "train": str(train_path),
                "val": str(val_path),
                "test": str(test_path),
            },
        },
        "majority": {},
        "stratified_random": {},
    }

    split_items = []
    if args.splits in ("val", "both"):
        split_items.append(("val", y_val))
    if args.splits in ("test", "both"):
        split_items.append(("test", y_test))

    for split_name, y in split_items:
        m = compute_metrics(y, majority_pred(y))
        results["majority"][split_name] = m.__dict__

        # Re-seed per split to make results deterministic and comparable
        rng = np.random.default_rng(args.seed + (0 if split_name == "val" else 1))
        sr_pred = rng.choice([0, 1], size=len(y), replace=True, p=[train_neg, train_pos]).astype(int)
        m2 = compute_metrics(y, sr_pred)
        results["stratified_random"][split_name] = m2.__dict__

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("Saved to:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
