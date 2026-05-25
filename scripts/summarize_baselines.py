import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _safe_get(d: dict[str, Any], path: list[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _add_row(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    rows.append(kwargs)


def parse_llm_metrics(rel_path: str, obj: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    meta = obj.get("meta")
    if not isinstance(meta, dict) or "model" not in meta:
        return False

    provider = meta.get("provider")
    if not provider:
        # Backward-compatible: infer from path.
        provider = "ollama" if "llm_ollama" in rel_path else "llm"

    for split in ("val", "test"):
        s = obj.get(split)
        if s is None:
            continue
        if not isinstance(s, dict):
            continue
        _add_row(
            rows,
            category=f"llm_{provider}",
            run=rel_path,
            method=str(meta.get("method") or f"{provider}_chat"),
            provider=provider,
            model=str(meta.get("model")),
            split=split,
            acc=s.get("acc"),
            f1=s.get("f1"),
            precision=s.get("precision"),
            recall=s.get("recall"),
            examples=s.get("examples"),
            seconds=s.get("seconds"),
            examples_per_second=s.get("examples_per_second"),
            parse_failures=s.get("parse_failures"),
            prompt_sha=meta.get("prompt_sha"),
            temperature=meta.get("temperature"),
            top_p=meta.get("top_p"),
            num_predict=meta.get("num_predict"),
            timeout_s=meta.get("timeout_s"),
            base_url=meta.get("base_url"),
        )
    return True


def parse_lower_metrics(rel_path: str, obj: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    if "majority" not in obj and "stratified_random" not in obj:
        return False

    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}

    for method in ("majority", "stratified_random"):
        mobj = obj.get(method)
        if not isinstance(mobj, dict):
            continue
        for split in ("val", "test"):
            s = mobj.get(split)
            if not isinstance(s, dict):
                continue
            _add_row(
                rows,
                category="lower",
                run=rel_path,
                method=method,
                model=None,
                split=split,
                acc=s.get("acc"),
                f1=s.get("f1"),
                precision=s.get("precision"),
                recall=s.get("recall"),
                examples=None,
                seconds=None,
                examples_per_second=None,
                parse_failures=None,
                prompt_sha=None,
                temperature=None,
                top_p=None,
                num_predict=None,
                timeout_s=None,
                seed=meta.get("seed"),
            )
    return True


def parse_tfidf_lr_metrics(rel_path: str, obj: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    meta = obj.get("meta")
    if not isinstance(meta, dict) or "best_C" not in meta:
        return False
    if not isinstance(obj.get("selection"), dict) or not isinstance(obj.get("final"), dict):
        return False

    best_val = _safe_get(obj, ["selection", "best_val"]) or _safe_get(obj, ["selection", "best", "val"])  # type: ignore[assignment]
    test = _safe_get(obj, ["final", "test"]) or _safe_get(obj, ["final", "test_metrics"])  # type: ignore[assignment]

    method = "tfidf_lr"
    analyzer = "word" if meta.get("use_word") else "char"
    method_detail = f"{method}({analyzer})"

    if isinstance(best_val, dict):
        _add_row(
            rows,
            category="tfidf_lr",
            run=rel_path,
            method=method_detail,
            model=None,
            split="val",
            acc=best_val.get("acc"),
            f1=best_val.get("f1"),
            precision=best_val.get("precision"),
            recall=best_val.get("recall"),
            examples=None,
            seconds=None,
            examples_per_second=None,
            parse_failures=None,
            prompt_sha=None,
            temperature=None,
            top_p=None,
            num_predict=None,
            timeout_s=None,
            best_C=meta.get("best_C"),
            min_df=meta.get("min_df"),
            seed=meta.get("seed"),
        )

    if isinstance(test, dict):
        _add_row(
            rows,
            category="tfidf_lr",
            run=rel_path,
            method=method_detail,
            model=None,
            split="test",
            acc=test.get("acc"),
            f1=test.get("f1"),
            precision=test.get("precision"),
            recall=test.get("recall"),
            examples=None,
            seconds=None,
            examples_per_second=None,
            parse_failures=None,
            prompt_sha=None,
            temperature=None,
            top_p=None,
            num_predict=None,
            timeout_s=None,
            best_C=meta.get("best_C"),
            min_df=meta.get("min_df"),
            seed=meta.get("seed"),
        )

    return True


def parse_esd_zh_metrics(rel_path: str, obj: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    meta = obj.get("meta")
    if not isinstance(meta, dict) or meta.get("category") != "esd_zh":
        return False

    method = str(meta.get("method") or "esd_zh")

    for split in ("val", "test"):
        s = obj.get(split)
        if s is None:
            continue
        if not isinstance(s, dict):
            continue
        _add_row(
            rows,
            category="esd_zh",
            run=rel_path,
            method=method,
            model=None,
            split=split,
            acc=s.get("acc"),
            f1=s.get("f1"),
            precision=s.get("precision"),
            recall=s.get("recall"),
            examples=s.get("examples"),
            seconds=s.get("seconds"),
            examples_per_second=s.get("examples_per_second"),
            seed=meta.get("seed"),
            emb_dim=meta.get("emb_dim"),
            hidden_dim=meta.get("hidden_dim"),
            attn_dim=meta.get("attn_dim"),
            dropout=meta.get("dropout"),
            max_len=meta.get("max_len"),
            vocab_size=meta.get("vocab_size"),
            best_epoch=meta.get("best_epoch"),
            train_seconds=meta.get("train_seconds"),
            threshold=s.get("threshold"),
            best_threshold=s.get("best_threshold"),
        )

    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize baseline metrics.json files into a single table.")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    ap.add_argument("--output_csv", type=str, default="artifacts/summary.csv")
    ap.add_argument("--output_json", type=str, default="artifacts/summary.json")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        raise FileNotFoundError(artifacts_dir)

    metric_paths = sorted(artifacts_dir.glob("**/metrics*.json"))

    rows: list[dict[str, Any]] = []
    unknown: list[str] = []

    for p in metric_paths:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            obj = json.loads(p.read_text(encoding="utf-8-sig"))

        rel_path = str(p.relative_to(artifacts_dir))

        if not isinstance(obj, dict):
            unknown.append(rel_path)
            continue

        parsed = False
        parsed = parsed or parse_llm_metrics(rel_path, obj, rows)
        parsed = parsed or parse_lower_metrics(rel_path, obj, rows)
        parsed = parsed or parse_tfidf_lr_metrics(rel_path, obj, rows)
        parsed = parsed or parse_esd_zh_metrics(rel_path, obj, rows)

        if not parsed:
            unknown.append(rel_path)

    rows.sort(key=lambda r: (str(r.get("category")), str(r.get("method")), str(r.get("model")), str(r.get("split"))))

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows, "unknown": unknown}, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Stable column order
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Found {len(metric_paths)} metric files")
    print(f"Wrote {len(rows)} rows")
    print("CSV:", out_csv)
    print("JSON:", out_json)
    if unknown:
        print("Unknown schemas:")
        for u in unknown:
            print(" -", u)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
