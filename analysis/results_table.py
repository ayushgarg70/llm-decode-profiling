"""Generate notes/RESULTS.md: every number the post may quote, straight from the CSVs, with run IDs.

    uv run python -m analysis.results_table                                   # post 1 (H100)
    uv run python -m analysis.results_table --gpu <name> --out notes/RESULTS_<name>.md   # another GPU

Sources: data/ceilings.json (E0), data/results.csv minus data/excluded_runs.csv (timing runs; H100 only),
data/kernels.csv (nsys breakdowns, latest analysis per trace), data/ppl.csv (perplexity).
Runs with the same configuration are pooled: the table shows their median, min and max of the per-run
median step time, and the run IDs. The floor and "% of floor" are recomputed here from the GPU's CURRENT
measured roof (data/ceilings*.json), not taken from the CSV, whose values used the roof known at run time
(the H100 roof was re-measured on 2026-11-18 with a better kernel: 3.08 -> 3.154 TB/s).
Never edit RESULTS.md by hand; regenerate it.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

OUT = Path("notes/RESULTS.md")
GPU = "H100"
KEY = ("experiment", "impl", "mode", "quant", "kv_dtype", "batch", "context_len", "fill")


def read(path: str) -> list[dict]:
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


LLAMA = {"model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336, "num_attention_heads": 32,
         "num_hidden_layers": 32, "num_key_value_heads": 8, "vocab_size": 128256, "tie_word_embeddings": False}


def floor_ms(r: dict, bandwidth: float) -> float:
    """Bandwidth floor for this run: weights as stored (quantized runs record weight_GB) + KV cache read."""
    from analysis.predict import ModelShape, kv_bytes_per_token, weight_bytes_per_step

    m = ModelShape.from_config(LLAMA)
    w = next((float(tok.split("=")[1]) * 1e9 for tok in r["notes"].split() if tok.startswith("weight_GB=")),
             weight_bytes_per_step(m))
    kv = kv_bytes_per_token(m, 1.0 if r["kv_dtype"] == "fp8" else 2.0)
    return (w + int(r["batch"]) * int(r["context_len"]) * kv) / bandwidth * 1e3


def timing_groups() -> dict[str, list[dict]]:
    from analysis.predict import ceilings_path, gpu_key

    bw = json.loads(Path(ceilings_path(gpu_key(GPU))).read_text())["bandwidth_Bps"]
    excluded = {r["run_id"] for r in read("data/excluded_runs.csv")}
    groups = defaultdict(list)
    for r in read("data/results.csv"):
        if r["run_id"] in excluded or GPU not in r["gpu"]:
            continue
        groups[tuple(r[k] for k in KEY)].append(r)
    by_exp = defaultdict(list)
    for key, rows in groups.items():
        ms = [float(r["ms_step_median"]) for r in rows]
        med = statistics.median(ms)
        b = int(rows[0]["batch"])
        by_exp[key[0]].append({
            **dict(zip(KEY, key)), "n": len(rows), "ms": med, "ms_min": min(ms), "ms_max": max(ms),
            "tok_s": b / med * 1e3, "pct_floor": 100 * floor_ms(rows[0], bw) / med,
            "floor_ms": floor_ms(rows[0], bw), "max_seq": rows[0]["max_seq"],
            "run_ids": ", ".join(r["run_id"] for r in rows),
            "notes": "; ".join(sorted({r["notes"] for r in rows if r["notes"]}))[:120]})
    return by_exp


def table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    out = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(format(r[c[0]], c[2]) if c[2] else str(r[c[0]]) for c in cols) + " |")
    return "\n".join(out)


TIMING_COLS = [("impl", "impl", ""), ("mode", "mode", ""), ("quant", "weights", ""), ("kv_dtype", "KV", ""),
               ("batch", "B", ""), ("context_len", "context", ""), ("fill", "fill", ""), ("n", "runs", ""),
               ("ms", "ms/step (median)", ".3f"), ("ms_min", "min", ".3f"), ("ms_max", "max", ".3f"),
               ("tok_s", "tok/s", ",.0f"), ("floor_ms", "floor ms", ".3f"), ("pct_floor", "% of floor", ".1f"),
               ("run_ids", "run IDs", "")]


def kernels_section() -> str:
    latest = {}
    for r in read("data/kernels.csv"):
        latest.setdefault(r["trace"], {})[r["group"]] = r  # later rows win
    out = []
    for trace, groups in latest.items():
        rows = [{"group": g, "ms": float(r["ms_per_step"]), "launches": float(r["launches_per_step"]),
                 "run_id": r["run_id"]} for g, r in groups.items()]
        out.append(f"**{trace}** (step time from `{rows[0]['run_id'] or 'n/a'}`)\n\n"
                   + table(sorted(rows, key=lambda r: -r["ms"]),
                           [("group", "group", ""), ("ms", "ms/step", ".3f"), ("launches", "launches/step", ".0f")]))
    return "\n\n".join(out)


STEP_TRACES = ["E1_own_eager_light", "E1_hf_loop", "E3_eager_graph", "E3_compiled", "E3_compiled_graph",
               "E4_compiled_graph_64k", "E6_int8_B1", "E6_int4_B1", "E6_fp8_B1"]


def short_kernel(name: str) -> str:
    """Kernel name without template arguments, plus the functor/impl that identifies an at::native kernel."""
    import re

    base = name.removeprefix("void ").split("<")[0].split("(")[0]
    tag = re.search(r"(\w+Functor|\w+_kernel_impl\w*)", name[len(base):])
    return f"{base} [{tag.group(1)}]" if tag else base


def steps_section() -> str:
    """Per-kernel detail of one traced decode step (data/steps/, extracted from the nsys traces)."""
    out = []
    for trace in STEP_TRACES:
        path = Path(f"data/steps/{trace}_step2.csv")
        if not path.exists():
            continue
        acc = defaultdict(list)
        for r in csv.DictReader(path.open()):
            acc[(r["group"], short_kernel(r["kernel"]), r["grid"])].append(float(r["dur_us"]))
        rows = [{"group": g, "kernel": k[:90], "grid": grid, "n": len(d), "us": sum(d) / len(d), "ms": sum(d) / 1e3}
                for (g, k, grid), d in acc.items()]
        n, ms = sum(r["n"] for r in rows), sum(r["ms"] for r in rows)
        out.append(f"**{trace}**, decode step 2: {n:,} kernels, {ms:.3f} ms of kernel time (traced)\n\n" + table(
            sorted(rows, key=lambda r: -r["ms"]),
            [("group", "group", ""), ("kernel", "kernel", ""), ("grid", "blocks", ""), ("n", "count", ""),
             ("us", "µs each", ".2f"), ("ms", "total ms", ".3f")]))
    return "\n\n".join(out)


def main() -> None:
    import argparse

    from analysis.predict import ceilings_path, gpu_key

    global GPU, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="H100", help="substring of the gpu column, e.g. H100 or B200")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    GPU, OUT = args.gpu, Path(args.out)
    parts = [f"# RESULTS for {GPU} (generated by `analysis/results_table.py`; do not edit)\n"]
    cpath = Path(ceilings_path(gpu_key(GPU)))
    c = json.loads(cpath.read_text()) if cpath.exists() else None
    if c:
        parts.append("## E0 ceilings\n\n" + table([{
            "sum": c["read_Bps"] / 1e12, "rows": c.get("read_rows_Bps", float("nan")) / 1e12, "copy": c["copy_Bps"] / 1e12,
            "roof": c["bandwidth_Bps"] / 1e12, "kernel": c.get("roof_kernel", "sum"), "tf": c["peak_flops"] / 1e12,
            "ridge": c["peak_flops"] / c["bandwidth_Bps"], "date": c["date"]}],
            [("sum", "read, x.sum() TB/s", ".3f"), ("rows", "read, row-sum TB/s", ".3f"), ("copy", "copy TB/s", ".3f"),
             ("roof", "roof TB/s", ".3f"), ("kernel", "roof kernel", ""), ("tf", "BF16 matmul TFLOP/s", ".1f"),
             ("ridge", "ridge FLOP/B", ".0f"), ("date", "run", "")]))
    by_exp = timing_groups()
    order = ["E1", "E1-recheck", "E3", "E4", "E5", "E6", "E7", "repeat", "B1", "B2", "B3", "B4"]
    for exp in order + sorted(set(by_exp) - set(order)):
        if exp not in by_exp or exp == "adhoc":
            continue
        rows = sorted(by_exp[exp], key=lambda r: (r["impl"], r["mode"], r["quant"], r["kv_dtype"],
                                                  int(r["context_len"]), int(r["batch"])))
        parts.append(f"## {exp}\n\n" + table(rows, TIMING_COLS))
    ppl = [r for r in read("data/ppl.csv") if GPU in r["gpu"]]
    if ppl:
        parts.append("## E6 perplexity (WikiText-2 test)\n\n" + table(ppl, [
            ("quant", "weights", ""), ("ppl", "perplexity", ""), ("tokens_scored", "tokens", ""),
            ("windows", "windows", ""), ("seq_len", "window", ""), ("weight_GB", "weight GB read/step", ""),
            ("date", "date", "")]))
    parts.append("## Kernel breakdowns (nsys; idle = timing-run step time - kernel time)\n\n" + kernels_section())
    if GPU == "H100":
        parts.append("## Per-kernel detail of one traced step (data/steps/, from the nsys traces; traced durations)\n\n" + steps_section())
    excluded = read("data/excluded_runs.csv")
    parts.append(f"## Excluded runs ({len(excluded)})\n\n" + table(excluded, [("run_id", "run ID", ""), ("reason", "reason", "")]))
    OUT.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
