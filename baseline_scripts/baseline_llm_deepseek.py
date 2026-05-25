import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


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


def read_xy(csv_path: Path, text_col: str, label_col: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if text_col not in df.columns:
        raise ValueError(f"Missing text_col '{text_col}' in {csv_path}. cols={list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label_col '{label_col}' in {csv_path}. cols={list(df.columns)}")

    x = (
        df[text_col]
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .fillna("")
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


def default_prompt() -> str:
    return (
        "你是一个二分类器。任务：判断一句中文短文本是否为反讽/讽刺。\n"
        "反讽=说反话、表面正面但实际负面/嘲讽、或通过语气/emoji表达相反态度。\n"
        "你必须结合“文本语义 + 语气（标点/重复/引号/夸张）+ Emoji（如🙂😂😏🙃等的语用）”综合判断；"
        "Emoji 可能强化或反转语气，但 Emoji 本身不等于反讽。\n"
        "只输出JSON：{\"irony\":0} 或 {\"irony\":1}，不要输出其它任何字符。"
    )


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def deepseek_chat(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
) -> tuple[str, dict]:
    """Call DeepSeek OpenAI-compatible chat completions API.

    Returns: (content, extra)
    """

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    content = ""
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = ""

    extra = {
        "http_url": url,
        "usage": data.get("usage"),
        "finish_reason": _safe_get(data, ["choices", 0, "finish_reason"]),
    }
    return content, extra


def _safe_get(obj, path):
    cur = obj
    for p in path:
        if isinstance(p, int):
            if not isinstance(cur, list) or p >= len(cur):
                return None
            cur = cur[p]
        else:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
    return cur


def parse_irony(raw: str) -> int | None:
    raw = (raw or "").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "irony" in obj:
            v = int(obj["irony"])
            if v in (0, 1):
                return v
    except Exception:
        pass

    for tok in ("0", "1"):
        if raw == tok:
            return int(tok)

    if "\"irony\"" in raw:
        if ":1" in raw or ": 1" in raw:
            return 1
        if ":0" in raw or ": 0" in raw:
            return 0

    return None


def eval_split(
    texts: list[str],
    labels: np.ndarray,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
    max_examples: int | None,
    sleep_ms: int,
    desc: str = "eval",
) -> tuple[Metrics, dict]:
    n = len(texts)
    if max_examples is not None:
        n = min(n, max_examples)

    preds: list[int] = []
    bad: list[dict] = []
    t0 = time.time()

    iterator = range(n)
    if tqdm is not None:
        iterator = tqdm(iterator, total=n, desc=desc, unit="ex")

    for i in iterator:
        raw, call_extra = deepseek_chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_text=f"文本：{texts[i]}\n输出：",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        pred = parse_irony(raw)
        if pred is None:
            bad.append({"idx": i, "text": texts[i], "raw": raw, "call": call_extra})
            pred = 0
        preds.append(pred)
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    dt = time.time() - t0
    y_true = labels[:n]
    y_pred = np.array(preds, dtype=int)
    m = compute_metrics(y_true, y_pred)

    extra = {
        "examples": int(n),
        "seconds": float(dt),
        "examples_per_second": float(n / dt) if dt > 0 else None,
        "parse_failures": int(len(bad)),
        "parse_failure_samples": bad[:5],
    }
    return m, extra


def main() -> int:
    parser = argparse.ArgumentParser(description="Strong baseline via DeepSeek API (OpenAI-compatible).")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="DeepSeek API key. If omitted, read from env DEEPSEEK_API_KEY.",
    )
    parser.add_argument("--val_csv", type=str, default="data/emoji_val.csv")
    parser.add_argument("--test_csv", type=str, default="data/emoji_test.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="irony")
    parser.add_argument(
        "--splits",
        type=str,
        default="both",
        choices=["val", "test", "both"],
        help="Which split(s) to evaluate. Use val while iterating prompts.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_predict", type=int, default=16, help="Mapped to max_tokens for DeepSeek")
    parser.add_argument("--timeout_s", type=int, default=120)
    parser.add_argument("--sleep_ms", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=None, help="For quick dry-run")
    parser.add_argument("--prompt", type=str, default=None, help="Optional system prompt override")
    parser.add_argument("--prompt_file", type=str, default=None, help="Read system prompt from a UTF-8 text file")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing DeepSeek API key. Provide --api_key or set env DEEPSEEK_API_KEY."
        )

    if args.prompt_file:
        system_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    else:
        system_prompt = args.prompt or default_prompt()

    val_path = Path(args.val_csv)
    test_path = Path(args.test_csv)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        safe_model = args.model.replace(":", "_").replace("/", "_")
        out_dir = Path("artifacts/baselines/llm_deepseek") / safe_model

    results_val = None
    results_test = None

    if args.splits in ("val", "both"):
        if not val_path.exists():
            raise FileNotFoundError(val_path)
        X_val, y_val = read_xy(val_path, args.text_col, args.label_col)
        val_m, val_extra = eval_split(
            texts=X_val,
            labels=y_val,
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            system_prompt=system_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.num_predict,
            timeout_s=args.timeout_s,
            max_examples=args.max_examples,
            sleep_ms=args.sleep_ms,
            desc="val",
        )
        results_val = {**val_m.__dict__, **val_extra}

    if args.splits in ("test", "both"):
        if not test_path.exists():
            raise FileNotFoundError(test_path)
        X_test, y_test = read_xy(test_path, args.text_col, args.label_col)
        test_m, test_extra = eval_split(
            texts=X_test,
            labels=y_test,
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            system_prompt=system_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.num_predict,
            timeout_s=args.timeout_s,
            max_examples=args.max_examples,
            sleep_ms=args.sleep_ms,
            desc="test",
        )
        results_test = {**test_m.__dict__, **test_extra}

    results = {
        "meta": {
            "provider": "deepseek",
            "model": args.model,
            "base_url": args.base_url,
            "splits": args.splits,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_predict": args.num_predict,
            "timeout_s": args.timeout_s,
            "sleep_ms": args.sleep_ms,
            "prompt_sha": prompt_hash(system_prompt),
        },
        "prompt": system_prompt,
        "val": results_val,
        "test": results_test,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("Saved to:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
