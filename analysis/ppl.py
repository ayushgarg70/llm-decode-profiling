"""Perplexity on WikiText-2 (raw, test split): the quality check that goes next to every E6 speed number.

Fixed, documented subset: the test split joined with "\\n\\n" (as in the HF perplexity guide), tokenized
once without special tokens, cut into non-overlapping windows of 2,047 tokens, each prefixed with BOS
(2,048 tokens per window). Every window is scored; the first --windows of them if given. Perplexity =
exp(mean negative log-likelihood over all predicted tokens). Same script and windows for every scheme.

    uv run python -m analysis.ppl --quant int4                # on the GPU box
    uv run python -m analysis.ppl --model tiny --device cpu --windows 2   # smoke test

Appends one row to data/ppl.csv (append-only).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import date
from pathlib import Path

import torch
import torch.nn.functional as F

PPL_CSV = Path("data/ppl.csv")
COLUMNS = ["date", "model", "quant", "seq_len", "windows", "tokens_scored", "nll", "ppl", "weight_GB", "gpu", "seconds", "notes"]


def wikitext2_test_ids(tokenizer) -> torch.Tensor:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    text = "\n\n".join(pd.read_parquet(path)["text"])
    return torch.tensor(tokenizer(text, add_special_tokens=False).input_ids)


def windows_of(ids: torch.Tensor, seq_len: int, bos: int | None, limit: int | None) -> list[torch.Tensor]:
    body = seq_len - (1 if bos is not None else 0)
    out = []
    for start in range(0, len(ids) - body + 1, body):
        w = ids[start:start + body]
        out.append(torch.cat([torch.tensor([bos]), w]) if bos is not None else w)
        if limit and len(out) == limit:
            break
    return out


@torch.no_grad()
def perplexity(model, windows: list[torch.Tensor], device) -> tuple[float, int]:
    """Mean NLL over all next-token predictions in the windows (positions 1..T-1 of each)."""
    seq_len = len(windows[0])
    model.setup_caches(1, seq_len, device=device)
    pos = torch.arange(seq_len, device=device)
    nll_sum, n = 0.0, 0
    for w in windows:
        x = w.to(device)[None]
        logits = model(x, pos).float()[0, :-1]
        nll_sum += F.cross_entropy(logits, x[0, 1:], reduction="sum").item()
        n += seq_len - 1
    return nll_sum / n, n


def main() -> None:
    from decode.quant import SCHEMES, quantize, weight_bytes_read_per_step
    from decode.run import DEFAULT_MODEL, build

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quant", choices=SCHEMES, default="bf16")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=None, help="score only the first N windows")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tok = build(args.model, device, dtype)
    quantize(model, args.quant)
    if tok is None:  # tiny model: random tokens, only checks the plumbing
        g = torch.Generator().manual_seed(0)
        ids, bos = torch.randint(0, model.args.vocab, (args.seq_len * 4,), generator=g), None
    else:
        ids, bos = wikitext2_test_ids(tok), tok.bos_token_id
    wins = windows_of(ids, args.seq_len, bos, args.windows)
    t0 = time.perf_counter()
    nll, n = perplexity(model, wins, device)
    row = {"date": date.today().isoformat(), "model": args.model, "quant": args.quant, "seq_len": args.seq_len,
           "windows": len(wins), "tokens_scored": n, "nll": f"{nll:.6f}", "ppl": f"{math.exp(nll):.4f}",
           "weight_GB": f"{weight_bytes_read_per_step(model) / 1e9:.4f}",
           "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
           "seconds": f"{time.perf_counter() - t0:.1f}", "notes": args.notes}
    print(row)
    if not args.no_log:
        new = not PPL_CSV.exists()
        with PPL_CSV.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
