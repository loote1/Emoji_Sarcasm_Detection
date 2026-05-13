#!/usr/bin/env python3
"""Serve the visualization frontend and run real ESD-ZH model inference."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent

try:
    import regex
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
except ModuleNotFoundError as exc:  # pragma: no cover - gives a friendly runtime message
    raise SystemExit(
        "缺少推理依赖，无法加载真实模型。请先运行：python3 -m pip install torch regex"
    ) from exc

EMOJI_RE = regex.compile(r"\p{Extended_Pictographic}")


def tokenize_zh_with_emoji(text: str) -> list[str]:
    text = (text or "").replace("\ufeff", "").strip()
    if not text:
        return []

    tokens: list[str] = []
    for grapheme in regex.findall(r"\X", text):
        if not grapheme.strip():
            continue
        prefix = "E:" if EMOJI_RE.search(grapheme) else "T:"
        tokens.append(prefix + grapheme)
    return tokens


def encode(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    unk = vocab["<UNK>"]
    ids = [vocab.get(token, unk) for token in tokens] or [unk]
    return ids[:max_len]


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
        out = self.dropout(out)

        u = torch.tanh(self.attn_fc(out))
        scores = torch.matmul(u, self.attn_context)
        _, tlen = scores.shape
        mask = torch.arange(tlen, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=1)
        v = torch.sum(out * alpha.unsqueeze(-1), dim=1)
        return self.out(v).squeeze(-1)


class Predictor:
    def __init__(self, model_path: Path, device: str) -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.vocab: dict[str, int] = checkpoint["vocab"]
        args = checkpoint.get("args", {})

        self.max_len = int(args.get("max_len", 160))
        self.threshold = float(args.get("threshold", 0.49))
        self.model = BiGRUWithAttention(
            vocab_size=len(self.vocab),
            pad_id=self.vocab["<PAD>"],
            emb_dim=int(args.get("emb_dim", 300)),
            hidden_dim=int(args.get("hidden_dim", 64)),
            attn_dim=int(args.get("attn_dim", 64)),
            dropout=float(args.get("dropout", 0.5)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        tokens = tokenize_zh_with_emoji(text)
        ids = encode(tokens, self.vocab, self.max_len)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)

        logits = self.model(x, lengths)
        probability = float(torch.sigmoid(logits)[0].detach().cpu().item())
        label = int(probability >= self.threshold)

        return {
            "label": label,
            "meaning": "讽刺 / 反话" if label else "非讽刺",
            "probability": probability,
            "tokens": tokens[: self.max_len],
            "source": "real_model",
        }


class Handler(SimpleHTTPRequestHandler):
    predictor: Predictor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/health":
            self._send_json({"ok": True, "model": "esd_zh_bigru_attn"})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(payload.get("text", "")).strip()
            if not text:
                raise ValueError("text 不能为空")
            result = self.predictor.predict(text)
        except Exception as exc:  # pragma: no cover - returns useful browser error
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json(result)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ESD-ZH frontend with real model inference.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default=str(ROOT / "artifacts" / "esd_zh" / "model.pt"))
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"找不到模型文件：{model_path}")

    Handler.predictor = Predictor(model_path=model_path, device=args.device)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"真实模型服务已启动：http://{args.host}:{args.port}/")
    print("接口：POST /predict")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
