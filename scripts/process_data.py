import argparse
import sys
from pathlib import Path

import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_original = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in lower_to_original:
            return lower_to_original[key]
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Keep only comment+irony columns, rename comment->text, and save as CSV."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input CSV path (e.g. emoji_comments.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output CSV path (e.g. emoji_comments.cleaned.csv)",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    # Try common encodings seen on Windows/Excel exports.
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(input_path, encoding=encoding)
            last_error = None
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
    if last_error is not None:
        print(f"Failed to read CSV: {last_error}", file=sys.stderr)
        return 3

    comment_col = _pick_column(df, ["comment", "text", "content"]) or _pick_column(
        df, ["Comment", "Text", "CONTENT"]
    )
    irony_col = _pick_column(df, ["irony", "label", "y"]) or _pick_column(
        df, ["Irony", "Label", "Y"]
    )

    if comment_col is None:
        print(
            f"Cannot find comment column. Existing columns: {list(df.columns)}",
            file=sys.stderr,
        )
        return 4
    if irony_col is None:
        print(
            f"Cannot find irony/label column. Existing columns: {list(df.columns)}",
            file=sys.stderr,
        )
        return 5

    out = df[[comment_col, irony_col]].copy()
    out = out.rename(columns={comment_col: "text", irony_col: "irony"})

    # Normalize text and label.
    out["text"] = out["text"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    out["irony"] = pd.to_numeric(out["irony"], errors="coerce").astype("Int64")

    # Drop rows with empty text or missing label.
    out = out[out["text"].ne("")]
    out = out.dropna(subset=["irony"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(
        f"Wrote {len(out)} rows to {output_path} with columns: {list(out.columns)}",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
