"""E1 baseline: Hugging Face transformers in eager mode, the way a typical user runs it.

Two measurements, each appended to data/results.csv (impl = "hf"):
  mode=generate  model.generate(), greedy. Decode-only time per token by the difference method:
                 (T(prefill + N+1 new tokens) - T(prefill + 1 new token)) / N, median of `--repeats`.
                 p10/p90 are left empty (this method gives one number per repeat, not per step).
  mode=loop      a manual loop over model(...) with past_key_values (HF's dynamic cache), timed per
                 step with CUDA events. The gap between the two is generate()'s own overhead.

    python -m decode.hf_baseline --experiment E1 --batch 1 --context 512
    python -m decode.hf_baseline --only loop --profile-steps 5     # under nsys
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import date

import torch

from bench.env import environment
from bench.results import append_row, new_run_id
from bench.telemetry import Telemetry
from bench.timing import summarize, time_steps
from decode.run import DEFAULT_MODEL, TINY_CONFIG, make_prompt


def build_hf(model_name: str, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

    if model_name == "tiny":
        torch.manual_seed(0)
        return LlamaForCausalLM(LlamaConfig(**TINY_CONFIG)).to(device=device, dtype=dtype).eval(), None
    from decode.load import resolve

    path = resolve(model_name)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    return model, AutoTokenizer.from_pretrained(path)


def elapsed_ms(fn, device) -> float:
    if device.type == "cuda":
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b)
    import time

    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


def floor_ms(model, batch, context):
    from analysis.predict import ModelShape, current_ceilings, floor_ms as f

    ceilings, _ = current_ceilings()
    return f(ModelShape.from_config(model.config.to_dict()), batch, context, ceilings["bandwidth_Bps"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5, help="generate() repeats per length")
    ap.add_argument("--only", choices=("generate", "loop"))
    ap.add_argument("--experiment", default="E1")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--profile-steps", type=int, default=0, help="loop only: profile K steps, no CSV row")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tok = build_hf(args.model, device, dtype)
    prompt = make_prompt(tok, args.batch, args.context, model.config.vocab_size, device)
    pad = tok.eos_token_id if tok is not None else 0
    base = {
        "date": date.today().isoformat(), "experiment": args.experiment, "impl": "hf", "quant": "bf16",
        "kv_dtype": "bf16" if dtype == torch.bfloat16 else str(dtype), "batch": args.batch,
        "context_len": args.context, "max_seq": "", "fill": "prefill", "new_tokens": args.steps,
        "warmup": args.warmup, "notes": args.notes,
    }
    pred = floor_ms(model, args.batch, args.context)
    results = []

    with torch.no_grad():
        if args.only in (None, "generate") and not args.profile_steps:
            def gen(n):
                model.generate(prompt, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                               pad_token_id=pad, attention_mask=torch.ones_like(prompt))

            gen(args.warmup)  # warm-up
            run_id = new_run_id(args.experiment, "hf", "generate", args.batch, args.context)
            with Telemetry(f"data/telemetry/{run_id}.csv") as tel:
                t1 = [elapsed_ms(lambda: gen(1), device) for _ in range(args.repeats)]
                tn = [elapsed_ms(lambda: gen(args.steps + 1), device) for _ in range(args.repeats)]
            per_tok = [(b - a) / args.steps for a, b in zip(sorted(t1), sorted(tn))]
            med = (statistics.median(tn) - statistics.median(t1)) / args.steps
            results.append({**base, "run_id": run_id, "mode": "generate", "ms_step_median": med,
                            "ms_step_p10": "", "ms_step_p90": "", "tok_per_s": args.batch / med * 1e3,
                            "pred_ms_floor": pred, "pct_of_floor": 100 * pred / med, **tel.summary(),
                            "notes": (args.notes + f" difference method; per-repeat ms/token "
                                      f"{min(per_tok):.3f}-{max(per_tok):.3f}").strip()})

        if args.only in (None, "loop"):
            out = model(prompt, use_cache=True)
            pkv = out.past_key_values
            cur = out.logits[:, -1:].argmax(-1)

            def step(_i=0):
                nonlocal cur, pkv
                o = model(cur, past_key_values=pkv, use_cache=True)
                pkv = o.past_key_values
                cur = o.logits[:, -1:].argmax(-1)

            for _ in range(args.warmup):
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
                print(f"profiled {args.profile_steps} HF loop steps")
                return
            run_id = new_run_id(args.experiment, "hf", "loop", args.batch, args.context)
            with Telemetry(f"data/telemetry/{run_id}.csv") as tel:
                ms = time_steps(step, args.steps, device)
            s = summarize(ms)
            results.append({**base, "run_id": run_id, "mode": "loop", **s,
                            "tok_per_s": args.batch / s["ms_step_median"] * 1e3, "pred_ms_floor": pred,
                            "pct_of_floor": 100 * pred / s["ms_step_median"], **tel.summary()})

    env = environment()
    for r in results:
        print(f"{r['mode']:>8}: {r['ms_step_median']:.3f} ms/step, {r['tok_per_s']:.1f} tok/s, "
              f"{r['pct_of_floor']:.1f}% of floor ({r['pred_ms_floor']:.3f} ms)")
        if not args.no_log:
            append_row({**r, **env})
    if results and not args.no_log:
        print("appended to data/results.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
