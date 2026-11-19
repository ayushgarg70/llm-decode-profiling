"""E7: vLLM as a production reference, measured the same way as our HF baseline.

Runs in its own venv (vLLM pins its own torch):  .venv-vllm/bin/python -m decode.vllm_bench ...
One process = one engine configuration (CUDA graphs or --enforce-eager, KV cache dtype, max length);
it then loops over batch sizes and contexts.

Decode time per step uses the difference method: T(N+1 new tokens) - T(1 new token), divided by N,
median over --repeats. Both runs do the same prefill, so it cancels. Prompts are random token ids (a
different seed per request and per repeat), prefix caching is off (so no prefill is skipped), and
min_tokens = max_tokens with ignore_eos so every request decodes exactly N+1 tokens. The prefill
token budget is raised to the whole batch so all prefill happens before the first decode step.

    .venv-vllm/bin/python -m decode.vllm_bench --batches 1 8 32 --contexts 2048
    .venv-vllm/bin/python -m decode.vllm_bench --enforce-eager --batches 1
    .venv-vllm/bin/python -m decode.vllm_bench --kv-cache-dtype fp8 --max-model-len 131072 --contexts 32768 130000

Appends one row per (batch, context) to data/results.csv with impl = "vllm".
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import date

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"


def floor_ms(model: str, batch: int, context: int, kv_bytes: float) -> float:
    from analysis.predict import ModelShape, floor_ms as fm, load_ceilings, load_config

    m = ModelShape.from_config(load_config(model))
    ceilings, _ = load_ceilings("data/ceilings.json", "H100-SXM")
    return fm(m, batch, context, ceilings["bandwidth_Bps"], kv_bytes=kv_bytes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batches", type=int, nargs="+", default=[1])
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048])
    ap.add_argument("--new", type=int, default=256, help="N: decode steps measured")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs, no torch.compile")
    ap.add_argument("--kv-cache-dtype", default="auto", help="auto (= BF16) or fp8")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--experiment", default="E7")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    max_b = max(args.batches)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              enforce_eager=args.enforce_eager, kv_cache_dtype=args.kv_cache_dtype, enable_prefix_caching=False,
              max_num_seqs=max(max_b, 1),
              max_num_batched_tokens=max(8192, max_b * (max(args.contexts) + 1)), seed=0)
    vocab = llm.get_tokenizer().vocab_size
    mode = "eager" if args.enforce_eager else "cuda_graphs"
    kv_dtype = "bf16" if args.kv_cache_dtype == "auto" else args.kv_cache_dtype

    def run(batch: int, context: int, n_new: int, seed: int) -> float:
        rng = random.Random(seed)
        prompts = [{"prompt_token_ids": [rng.randrange(1000, vocab) for _ in range(context)]} for _ in range(batch)]
        sp = SamplingParams(temperature=0.0, max_tokens=n_new, min_tokens=n_new, ignore_eos=True)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        assert all(len(o.outputs[0].token_ids) == n_new for o in outs), "wrong number of generated tokens"
        return dt

    from bench.env import environment
    from bench.results import append_row, new_run_id
    from bench.telemetry import Telemetry

    for context in args.contexts:
        for batch in args.batches:
            run(batch, context, args.new + 1, seed=12345)  # warm-up (allocations, graph shapes)
            run_id = new_run_id(args.experiment, "vllm", mode, batch, context)
            per_step = []
            with Telemetry(f"data/telemetry/{run_id}.csv") as tel:
                for r in range(args.repeats):
                    t_short = run(batch, context, 1, seed=1000 + 2 * r)
                    t_long = run(batch, context, args.new + 1, seed=1001 + 2 * r)
                    per_step.append((t_long - t_short) / args.new * 1e3)
            per_step.sort()
            med = statistics.median(per_step)
            pred = floor_ms(args.model, batch, context, kv_bytes=1.0 if kv_dtype == "fp8" else 2.0)
            row = {"run_id": run_id, "date": date.today().isoformat(), "experiment": args.experiment,
                   "impl": "vllm", "mode": mode, "quant": "bf16", "kv_dtype": kv_dtype, "batch": batch,
                   "context_len": context, "max_seq": args.max_model_len, "fill": "prefill",
                   "new_tokens": args.new, "warmup": 1, "ms_step_median": med,
                   "ms_step_p10": per_step[0], "ms_step_p90": per_step[-1],
                   "tok_per_s": batch / med * 1e3, "pred_ms_floor": pred, "pct_of_floor": 100 * pred / med,
                   **tel.summary(), **environment(),
                   "notes": f"vllm {vllm.__version__}; difference method, {args.repeats} repeats (p10/p90 = min/max); "
                            + args.notes}
            print(json.dumps({k: row[k] for k in ("run_id", "batch", "context_len", "kv_dtype", "ms_step_median",
                                                  "tok_per_s", "pct_of_floor")}, default=str))
            if not args.no_log:
                append_row(row)


if __name__ == "__main__":
    main()
