"""Time (or profile) decode steps of our own loop. One invocation = one configuration = one CSV row.

Examples:
    # CPU smoke test with a tiny random model (no GPU, no download):
    uv run python -m decode.run --model tiny --device cpu --context 32 --steps 8 --warmup 2 --no-log
    # E1: eager, batch 1, 512-token real prompt, 256 timed steps:
    python -m decode.run --experiment E1 --mode eager --batch 1 --context 512
    # E3: CUDA graphs via torch.compile:
    python -m decode.run --experiment E3 --mode compiled_graph --batch 1 --context 512
    # Profiling run (under nsys --capture-range=cudaProfilerApi): no CSV row, K steps captured:
    python -m decode.run --mode eager --profile-steps 5 --nvtx-layers

Modes (the E3 2x2):
    eager           plain PyTorch, one kernel launch per op
    eager_graph     the same eager step, captured once into a CUDA graph and replayed (no fusion)
    compiled        torch.compile(mode="default"): kernels fused, still launched one by one
    compiled_graph  torch.compile(mode="reduce-overhead"): fused AND replayed as a CUDA graph
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import torch

from bench.env import environment
from bench.results import append_row, new_run_id
from decode.quant import SCHEMES, quantize, weight_bytes_read_per_step
from bench.telemetry import Telemetry
from bench.timing import summarize, time_steps
from decode.generate import decode_one_token, prefill
from decode.model import ModelArgs, Transformer

MODES = ("eager", "eager_graph", "compiled", "compiled_graph")
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
PROMPT_FILE = Path(__file__).with_name("prompt.txt")

TINY_CONFIG = dict(  # same structure as Llama-3.1, tiny sizes: for CPU smoke tests only
    model_type="llama", hidden_size=64, intermediate_size=128, num_attention_heads=4,
    num_key_value_heads=2, num_hidden_layers=2, vocab_size=256, max_position_embeddings=4096,
    rms_norm_eps=1e-5, rope_theta=500000.0, tie_word_embeddings=False, attention_bias=False,
)


def build(model_name: str, device: torch.device, dtype: torch.dtype):
    """Returns (model, tokenizer or None)."""
    if model_name == "tiny":
        torch.manual_seed(0)
        return Transformer(ModelArgs.from_hf_config(TINY_CONFIG)).to(device=device, dtype=dtype).eval(), None
    from transformers import AutoTokenizer

    from decode.load import load_model, resolve

    path = resolve(model_name)
    return load_model(str(path), device=device, dtype=dtype), AutoTokenizer.from_pretrained(path)


def make_prompt(tokenizer, batch: int, length: int, vocab: int, device) -> torch.Tensor:
    """A fixed, real-text prompt of exactly `length` tokens (the text is repeated if needed), the same
    for every request in the batch. Random token ids for the tiny model."""
    if tokenizer is None:
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, vocab, (length,), generator=g)
    else:
        text_ids = tokenizer(PROMPT_FILE.read_text(), add_special_tokens=False).input_ids
        ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        while len(ids) < length:
            ids += text_ids
        ids = torch.tensor(ids[:length])
    return ids.to(device).expand(batch, length).contiguous()


def make_step_fn(mode: str, model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor):
    """Return step_fn(model, x, input_pos) -> next token [B, 1] for the given mode."""
    if mode == "eager":
        return decode_one_token
    if mode in ("compiled", "compiled_graph"):
        return torch.compile(decode_one_token, fullgraph=True,
                             mode="reduce-overhead" if mode == "compiled_graph" else "default")
    if mode == "eager_graph":
        if cur_token.device.type != "cuda":
            raise SystemExit("eager_graph needs CUDA")
        static_x = cur_token.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm up on a side stream before capture (CUDA graph rule)
            for _ in range(3):
                decode_one_token(model, static_x, input_pos)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = decode_one_token(model, static_x, input_pos)

        def step(_model, x, pos):
            assert pos is input_pos, "the graph was captured with this exact input_pos tensor"
            static_x.copy_(x)
            graph.replay()
            return static_out

        return step
    raise ValueError(mode)


def predicted_floor_ms(model: Transformer, batch: int, context: int, ceilings_path: str = "data/ceilings.json",
                       weight_bytes: float | None = None):
    """Bandwidth floor. BF16: from the config (analysis/predict.py). Quantized: the bytes the quantized
    weights actually occupy (values + scales), plus the same BF16 KV cache."""
    from analysis.predict import ModelShape, floor_ms, kv_bytes_per_token, load_ceilings

    m = ModelShape.from_config(model.args.hf_config)
    ceilings, _ = load_ceilings(ceilings_path, "H100-SXM")
    if weight_bytes is None:
        return floor_ms(m, batch, context, ceilings["bandwidth_Bps"])
    return 1e3 * (weight_bytes + batch * context * kv_bytes_per_token(m)) / ceilings["bandwidth_Bps"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id, local dir, or 'tiny'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=MODES, default="eager")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--context", type=int, default=512, help="tokens already in each request's KV cache")
    ap.add_argument("--fill", choices=("prefill", "synthetic"), default="prefill",
                    help="prefill a real prompt, or start decoding at position `context` over a zero "
                         "KV cache (same bytes read; used for long contexts)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=256, help="timed decode steps")
    ap.add_argument("--experiment", default="adhoc")
    ap.add_argument("--quant", choices=SCHEMES, default="bf16",
                    help="weight format (decode/quant.py); int8/int4 weight-only, fp8 weights + activations")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-log", action="store_true", help="don't append to data/results.csv")
    ap.add_argument("--profile-steps", type=int, default=0,
                    help="profile K steps between cudaProfilerStart/Stop (for nsys), then exit; no CSV row")
    ap.add_argument("--nvtx-layers", action="store_true", help="per-layer NVTX ranges (eager modes only)")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tokenizer = build(args.model, device, dtype)
    quantized = quantize(model, args.quant)
    weight_bytes = weight_bytes_read_per_step(model)
    if quantized:
        print(f"quantized {len(quantized)} linear layers to {args.quant} (incl. lm_head: {'lm_head' in quantized}); "
              f"weights read per step: {weight_bytes / 1e9:.3f} GB")

    n_after = args.warmup + max(args.steps, args.profile_steps) + 2
    max_seq = args.context + n_after
    if max_seq > model.args.max_position:
        raise SystemExit(f"context + steps = {max_seq} > max_position_embeddings {model.args.max_position}")
    model.setup_caches(args.batch, max_seq, device=device, dtype=dtype)

    with torch.no_grad():
        if args.fill == "prefill":
            first = prefill(model, make_prompt(tokenizer, args.batch, args.context, model.args.vocab, device))
        else:
            first = torch.randint(0, model.args.vocab, (args.batch, 1), device=device)
        input_pos = torch.tensor([args.context], device=device)
        step_fn = make_step_fn(args.mode, model, first, input_pos)
        model.nvtx_layers = args.nvtx_layers and args.mode == "eager" and args.profile_steps > 0
        cur = first

        def step(_i=0):
            nonlocal cur
            cur = step_fn(model, cur, input_pos).clone()
            input_pos.add_(1)

        for _ in range(args.warmup):  # includes torch.compile / graph recording
            step()

        if args.profile_steps:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            for i in range(args.profile_steps):
                torch.cuda.nvtx.range_push(f"decode step {i}")
                step()
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            print(f"profiled {args.profile_steps} steps ({args.mode}, B={args.batch}, s={args.context})")
            return

        run_id = new_run_id(args.experiment, "own" if args.quant == "bf16" else f"own-{args.quant}", args.mode, args.batch, args.context)
        with Telemetry(f"data/telemetry/{run_id}.csv") as tel:
            ms = time_steps(step, args.steps, device)

    s = summarize(ms)
    pred = predicted_floor_ms(model, args.batch, args.context, weight_bytes=weight_bytes if quantized else None)
    row = {
        "run_id": run_id, "date": date.today().isoformat(), "experiment": args.experiment,
        "impl": "own", "mode": args.mode, "quant": args.quant, "kv_dtype": "bf16" if dtype == torch.bfloat16 else str(dtype),
        "batch": args.batch, "context_len": args.context, "max_seq": max_seq, "fill": args.fill,
        "new_tokens": args.steps, "warmup": args.warmup, **s,
        "tok_per_s": args.batch / s["ms_step_median"] * 1e3,
        "pred_ms_floor": pred, "pct_of_floor": 100 * pred / s["ms_step_median"],
        **tel.summary(), **environment(),
        "notes": (args.notes + " " if args.notes else "") + (f"weight_GB={weight_bytes / 1e9:.4f}" if quantized else ""),
    }
    print(json.dumps({k: row[k] for k in ("run_id", "mode", "batch", "context_len", "ms_step_median",
                                          "ms_step_p10", "ms_step_p90", "tok_per_s", "pred_ms_floor",
                                          "pct_of_floor")}, indent=1, default=str))
    if not args.no_log:
        append_row(row)
        print("appended to data/results.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
