import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _stats(df: pd.DataFrame, label_col: str) -> dict:
    vc = df[label_col].value_counts(dropna=False).to_dict()
    return {
        "rows": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in vc.items()},
        "pos": int((df[label_col] == 1).sum()),
        "neg": int((df[label_col] == 0).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Split emoji irony dataset into train/val/test.")
    parser.add_argument("--input", type=str, default="data/emoji_comments.cleaned.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="irony")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--dedup_text", action="store_true", help="Drop duplicate text rows before splitting")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--prefix", type=str, default="emoji")
    args = parser.parse_args()

    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Ratios must sum to 1.0, got {total}")

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    for col in (args.text_col, args.label_col):
        if col not in df.columns:
            raise SystemExit(f"Missing column '{col}'. Columns={list(df.columns)}")

    df = df[[args.text_col, args.label_col]].copy()
    df[args.text_col] = df[args.text_col].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    df = df[df[args.text_col].ne("")]
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    df = df.dropna(subset=[args.label_col])
    df[args.label_col] = df[args.label_col].astype(int)

    df = df[df[args.label_col].isin([0, 1])]

    if args.dedup_text:
        df = df.drop_duplicates(subset=[args.text_col], keep="first")

    y = df[args.label_col]

    # First split out test
    train_val_df, test_df = train_test_split(
        df,
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=y,
    )

    # Then split train vs val from the remaining pool
    remaining = 1.0 - args.test_ratio
    val_size_in_train_val = args.val_ratio / remaining

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_in_train_val,
        random_state=args.seed,
        stratify=train_val_df[args.label_col],
    )

    out_dir = Path(args.output_dir)
    train_path = out_dir / f"{args.prefix}_train.csv"
    val_path = out_dir / f"{args.prefix}_val.csv"
    test_path = out_dir / f"{args.prefix}_test.csv"

    _save_csv(train_df, train_path)
    _save_csv(val_df, val_path)
    _save_csv(test_df, test_path)

    split_stats = {
        "input": str(in_path),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "dedup_text": bool(args.dedup_text),
        "train": _stats(train_df, args.label_col),
        "val": _stats(val_df, args.label_col),
        "test": _stats(test_df, args.label_col),
    }

    stats_path = out_dir / f"{args.prefix}_split_stats.json"
    stats_path.write_text(json.dumps(split_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wrote:")
    print("-", train_path)
    print("-", val_path)
    print("-", test_path)
    print("-", stats_path)
    print(json.dumps(split_stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
