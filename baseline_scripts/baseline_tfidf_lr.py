import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion


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


def read_xy(csv_path: Path, text_col: str, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if text_col not in df.columns:
        raise ValueError(f"Missing text_col '{text_col}' in {csv_path}. cols={list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label_col '{label_col}' in {csv_path}. cols={list(df.columns)}")

    x = df[text_col].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    x = x.fillna("")

    y = pd.to_numeric(df[label_col], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Found NaN labels in {csv_path}")
    y = y.astype(int).to_numpy()

    bad = set(np.unique(y)) - {0, 1}
    if bad:
        raise ValueError(f"Label must be 0/1, found {sorted(bad)} in {csv_path}")

    return x.to_numpy(), y


def build_vectorizer(min_df: int = 2, use_word: bool = False):
    # Char ngrams are the standard strong-weak baseline for Chinese.
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=min_df,
        max_features=300_000,
        sublinear_tf=True,
    )

    if not use_word:
        return char_vec

    # Word ngrams require tokenization; without it, this can be empty for pure Chinese text.
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=200_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\\b\\w+\\b",
    )
    return FeatureUnion([("char", char_vec), ("word", word_vec)])


def fit_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    C: float,
    seed: int,
    min_df: int,
    use_word: bool,
) -> tuple[Metrics, object, LogisticRegression]:
    vec = build_vectorizer(min_df=min_df, use_word=use_word)
    Xtr = vec.fit_transform(X_train)
    Xev = vec.transform(X_eval)

    clf = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
        class_weight=None,
    )
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xev)
    return compute_metrics(y_eval, pred), vec, clf


def main() -> int:
    parser = argparse.ArgumentParser(description="Weak baseline: TF-IDF (char/word ngrams) + LogisticRegression")
    parser.add_argument("--train_csv", type=str, default="data/emoji_train.csv")
    parser.add_argument("--val_csv", type=str, default="data/emoji_val.csv")
    parser.add_argument("--test_csv", type=str, default="data/emoji_test.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="irony")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--C_list", type=str, default="0.25,0.5,1,2,4")
    parser.add_argument("--use_word", action="store_true", help="Also add word-level TF-IDF (requires tokenization)")
    parser.add_argument(
        "--splits",
        type=str,
        default="both",
        choices=["val", "test", "both"],
        help="Which split(s) to evaluate.",
    )
    parser.add_argument("--output_dir", type=str, default="artifacts/baselines/tfidf_lr")
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    val_path = Path(args.val_csv)
    test_path = Path(args.test_csv)
    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(p)

    X_train, y_train = read_xy(train_path, args.text_col, args.label_col)
    X_val, y_val = read_xy(val_path, args.text_col, args.label_col)
    X_test, y_test = read_xy(test_path, args.text_col, args.label_col)

    C_list = [float(x.strip()) for x in args.C_list.split(",") if x.strip()]
    if not C_list:
        raise SystemExit("C_list is empty")

    # Select C on val by F1
    candidates = []
    best = None
    for C in C_list:
        m_val, _, _ = fit_and_eval(
            X_train,
            y_train,
            X_val,
            y_val,
            C=C,
            seed=args.seed,
            min_df=args.min_df,
            use_word=args.use_word,
        )
        row = {"C": C, **m_val.__dict__}
        candidates.append(row)
        if best is None or (m_val.f1 > best[0].f1):
            best = (m_val, C)

    assert best is not None
    best_val_metrics, best_C = best

    test_metrics = None
    if args.splits in ("test", "both"):
        # Refit on train+val with best C, then evaluate on test.
        X_trainval = np.concatenate([X_train, X_val], axis=0)
        y_trainval = np.concatenate([y_train, y_val], axis=0)

        vec = build_vectorizer(min_df=args.min_df, use_word=args.use_word)
        Xtr = vec.fit_transform(X_trainval)
        Xte = vec.transform(X_test)

        clf = LogisticRegression(
            C=best_C,
            solver="liblinear",
            max_iter=2000,
            random_state=args.seed,
            class_weight=None,
        )
        clf.fit(Xtr, y_trainval)
        test_pred = clf.predict(Xte)
        test_metrics = compute_metrics(y_test, test_pred)

    results = {
        "meta": {
            "seed": args.seed,
            "min_df": args.min_df,
            "use_word": bool(args.use_word),
            "C_list": C_list,
            "best_C": best_C,
            "splits": {
                "train": str(train_path),
                "val": str(val_path),
                "test": str(test_path),
            },
        },
        "selection": {
            "val_by_C": candidates,
            "best_val": best_val_metrics.__dict__,
        },
        "final": {
            "trainval_fit": True,
            "test": test_metrics.__dict__ if test_metrics is not None else None,
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("Saved to:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
