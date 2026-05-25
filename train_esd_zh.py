import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import regex
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


EMOJI_RE = regex.compile(r"\p{Extended_Pictographic}")


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


def tokenize_zh_with_emoji(text: str) -> list[str]:
    """Tokenize into a mixed sequence preserving order.

    - Uses Unicode grapheme clusters (\\X) to keep composite emojis intact.
    - Prefix tokens with 'E:' (emoji) or 'T:' (text) to avoid collisions.

    This is a pragmatic Chinese adaptation: non-emoji is treated roughly as
    character-level tokens (including punctuation).
    """

    text = (text or "").replace("\ufeff", "").strip()
    if not text:
        return []

    tokens: list[str] = []
    for g in regex.findall(r"\X", text):
        if not g.strip():
            continue
        if EMOJI_RE.search(g):
            tokens.append("E:" + g)
        else:
            tokens.append("T:" + g)
    return tokens


def build_vocab(
    tokenized_texts: list[list[str]],
    min_freq: int,
    max_vocab: int,
) -> dict[str, int]:
    if min_freq < 1:
        raise ValueError("min_freq must be >= 1")
    if max_vocab < 10:
        raise ValueError("max_vocab must be >= 10")

    counter: Counter[str] = Counter()
    for toks in tokenized_texts:
        counter.update(toks)

    # Stable ordering: frequency desc then token asc
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for tok, freq in items:
        if freq < min_freq:
            break
        if tok in vocab:
            continue
        if len(vocab) >= max_vocab:
            break
        vocab[tok] = len(vocab)

    return vocab


def encode(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    unk = vocab["<UNK>"]
    ids = [vocab.get(t, unk) for t in tokens]
    if not ids:
        ids = [unk]
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids


class EmojiSarcasmDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        vocab: dict[str, int],
        max_len: int,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels length mismatch")
        self._texts = texts
        self._labels = labels.astype(np.int64)
        self._vocab = vocab
        self._max_len = max_len

        self._encoded: list[list[int]] = []
        self._lengths: list[int] = []
        for t in texts:
            toks = tokenize_zh_with_emoji(t)
            ids = encode(toks, vocab, max_len)
            self._encoded.append(ids)
            self._lengths.append(len(ids))

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int):
        return self._encoded[idx], self._lengths[idx], int(self._labels[idx])


def make_collate_fn(pad_id: int):
    def _collate(batch):
        ids_list, lengths, labels = zip(*batch)
        max_len = int(max(lengths))

        x = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        lengths_t = torch.tensor(lengths, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.float32)
        return x, lengths_t, y

    return _collate


class BiGRUWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        emb_dim: int,
        hidden_dim: int,
        attn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.attn_fc = nn.Linear(hidden_dim * 2, attn_dim, bias=True)
        self.attn_context = nn.Parameter(torch.empty(attn_dim))
        nn.init.normal_(self.attn_context, mean=0.0, std=0.02)

        self.out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T), lengths: (B,)
        emb = self.dropout(self.embedding(input_ids))

        packed = pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=input_ids.size(1),
        )
        out = self.dropout(out)  # (B, T, 2H)

        # Attention: u_i = tanh(W h_i + b), score_i = u_i^T u_w
        u = torch.tanh(self.attn_fc(out))  # (B, T, A)
        scores = torch.matmul(u, self.attn_context)  # (B, T)

        bsz, tlen = scores.shape
        mask = torch.arange(tlen, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=1)  # (B, T)

        v = torch.sum(out * alpha.unsqueeze(-1), dim=1)  # (B, 2H)
        logits = self.out(v).squeeze(-1)  # (B,)
        return logits


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

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="train", unit="batch")

    for x, lengths, y in iterator:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        logits = model(x, lengths)
        loss = criterion(logits, y)

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
) -> tuple[Metrics, dict]:
    model.eval()

    all_logits: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []

    t0 = time.time()
    for x, lengths, y in loader:
        x = x.to(device)
        lengths = lengths.to(device)

        logits = model(x, lengths)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    dt = time.time() - t0

    logits = torch.cat(all_logits, dim=0) if all_logits else torch.zeros((0,))
    y_true = torch.cat(all_y, dim=0).numpy().astype(int) if all_y else np.zeros((0,), dtype=int)

    probs = torch.sigmoid(logits).numpy()
    y_pred = (probs >= 0.49).astype(int)

    m = compute_metrics(y_true, y_pred)
    extra = {
        "examples": int(len(y_true)),
        "seconds": float(dt),
        "examples_per_second": float(len(y_true) / dt) if dt > 0 else None,
    }
    return m, extra


def safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_+=.") else "_" for c in s)


def main() -> int:
    ap = argparse.ArgumentParser(description="ESD-ZH: Chinese adaptation of emoji+text fusion with BiGRU+Attention")
    ap.add_argument("--train_csv", type=str, default="data/emoji_train.csv")
    ap.add_argument("--val_csv", type=str, default="data/emoji_val.csv")
    ap.add_argument("--test_csv", type=str, default="data/emoji_test.csv")
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--label_col", type=str, default="irony")
    ap.add_argument(
        "--splits",
        type=str,
        default="val",
        choices=["val", "test", "both"],
        help="Which split(s) to report after training. (Early-stopping always uses val.)",
    )

    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--max_vocab", type=int, default=50000)

    ap.add_argument("--emb_dim", type=int, default=300)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--attn_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda/cpu/cuda:0... If omitted, auto-detect.",
    )

    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    if args.max_len < 1:
        raise SystemExit("max_len must be >= 1")

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

    tokenized_train = [tokenize_zh_with_emoji(t) for t in X_train]
    vocab = build_vocab(tokenized_train, min_freq=args.min_freq, max_vocab=args.max_vocab)

    pad_id = vocab["<PAD>"]

    ds_train = EmojiSarcasmDataset(X_train, y_train, vocab=vocab, max_len=args.max_len)
    ds_val = EmojiSarcasmDataset(X_val, y_val, vocab=vocab, max_len=args.max_len)
    ds_test = EmojiSarcasmDataset(X_test, y_test, vocab=vocab, max_len=args.max_len)

    collate_fn = make_collate_fn(pad_id=pad_id)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device_str = _default_device(args.device)
    device = torch.device(device_str)

    model = BiGRUWithAttention(
        vocab_size=len(vocab),
        pad_id=pad_id,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        attn_dim=args.attn_dim,
        dropout=args.dropout,
    ).to(device)

    # Handle imbalance (pos is minority)
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_weight = (neg / max(pos, 1.0)) if pos > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_epoch = 0
    best_state: dict | None = None

    history: list[dict] = []
    t_train0 = time.time()

    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=dl_train,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_m, _ = eval_split(model, dl_val, device=device)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_acc": float(val_m.acc),
            "val_f1": float(val_m.f1),
            "val_precision": float(val_m.precision),
            "val_recall": float(val_m.recall),
        }
        history.append(row)

        improved = val_m.f1 > best_f1 + 1e-6
        if improved:
            best_f1 = float(val_m.f1)
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if tqdm is None:
            print(
                f"epoch={epoch} loss={train_loss:.4f} val_f1={val_m.f1:.4f} val_acc={val_m.acc:.4f}"
            )

        if epochs_no_improve >= args.patience:
            break

    train_seconds = float(time.time() - t_train0)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Output directory
    run_name = args.run_name
    if not run_name:
        run_name = f"ed{args.emb_dim}_hd{args.hidden_dim}_ml{args.max_len}_s{args.seed}"
    run_name = safe_name(run_name)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("artifacts/esd_zh") / run_name

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model + vocab for reproducibility
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_cpu, "vocab": vocab, "args": vars(args)}, out_dir / "model.pt")
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    results_val = None
    results_test = None

    if args.splits in ("val", "both"):
        m, extra = eval_split(model, dl_val, device=device)
        results_val = {**m.__dict__, **extra}

    if args.splits in ("test", "both"):
        m, extra = eval_split(model, dl_test, device=device)
        results_test = {**m.__dict__, **extra}

    results = {
        "meta": {
            "category": "esd_zh",
            "method": "esd_zh_bigru_attn",
            "tokenizer": "regex\\X+Extended_Pictographic",
            "splits": args.splits,
            "seed": args.seed,
            "max_len": args.max_len,
            "min_freq": args.min_freq,
            "max_vocab": args.max_vocab,
            "vocab_size": int(len(vocab)),
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
            "attn_dim": args.attn_dim,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "best_epoch": int(best_epoch),
            "train_seconds": train_seconds,
            "pos_weight": float(pos_weight),
            "device": str(device),
            "paths": {
                "train": str(train_path),
                "val": str(val_path),
                "test": str(test_path),
            },
        },
        "history": history,
        "val": results_val,
        "test": results_test,
    }

    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("Saved to:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
