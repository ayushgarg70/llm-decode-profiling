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
from collections import Counter, defaultdict
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
        if trace.startswith("B") != (GPU == "B200"):
            continue
        rows = [{"group": g, "ms": float(r["ms_per_step"]), "launches": float(r["launches_per_step"]),
                 "run_id": r["run_id"]} for g, r in groups.items()]
        out.append(f"**{trace}** (step time from `{rows[0]['run_id'] or 'n/a'}`)\n\n"
                   + table(sorted(rows, key=lambda r: -r["ms"]),
                           [("group", "group", ""), ("ms", "ms/step", ".3f"), ("launches", "launches/step", ".0f")]))
    return "\n\n".join(out)


STEP_TRACES = {
    "H100": ["E1_own_eager_light", "E1_hf_loop", "E3_eager_graph", "E3_compiled", "E3_compiled_graph",
             "E4_compiled_graph_64k", "E6_int8_B1", "E6_int4_B1", "E6_fp8_B1"],
    "B200": ["B1_own_eager", "B1_eager_graph", "B1_compiled", "B1_compiled_graph", "B3_fp8_B1", "B3_nvfp4_B1",
             "B3_nvfp4w_B1", "B3_fp8_B32", "B3_nvfp4_B32"],
}
# Weight matmuls in the order they run in each layer (then lm_head once), and their BF16 weight bytes.
H, KV, FF, VOCAB, LAYERS = 4096, 1024, 14336, 128256, 32
ROLE_ORDER = ["q", "k", "v", "o", "gate", "up", "down"]
ROLE_BYTES = {"q": H * H * 2, "k": KV * H * 2, "v": KV * H * 2, "o": H * H * 2, "gate": FF * H * 2, "up": FF * H * 2,
              "down": H * FF * 2, "lm_head": VOCAB * H * 2}


def step_rows(trace: str) -> list[dict]:
    return list(csv.DictReader(Path(f"data/steps/{trace}_step2.csv").open()))


def is_matmul(r: dict) -> bool:
    """A weight-matmul kernel itself: cuBLAS (nvjet/gemm) or CUTLASS, not split-K reductions or quantization."""
    k = r["kernel"]
    return r["group"] == "weight_gemm" and ("nvjet" in k or "cutlass" in k or "gemm" in k.lower()) and "splitKreduce" not in k


def short_kernel(name: str) -> str:
    """Kernel name without template arguments, plus the functor/impl that identifies an at::native kernel."""
    import re

    base = name.removeprefix("void ").split("<")[0].split("(")[0]
    tag = re.search(r"(\w+Functor|\w+_kernel_impl\w*)", name[len(base):])
    return f"{base} [{tag.group(1)}]" if tag else base


def steps_section() -> str:
    """Per-kernel detail of one traced decode step (data/steps/, extracted from the nsys traces)."""
    out = []
    for trace in STEP_TRACES.get(GPU, []):
        path = Path(f"data/steps/{trace}_step2.csv")
        if not path.exists():
            continue
        acc = defaultdict(list)
        for r in csv.DictReader(path.open()):
            acc[(r["group"], short_kernel(r["kernel"]), r["grid"])].append(float(r["dur_us"]))
        rows = [{"group": g, "kernel": k[:90], "grid": grid, "n": len(d), "us": sum(d) / len(d), "ms": sum(d) / 1e3}
                for (g, k, grid), d in acc.items()]
        n, ms = sum(r["n"] for r in rows), sum(r["ms"] for r in rows)
        raw = step_rows(trace)
        mm = sum(float(r["dur_us"]) for r in raw if is_matmul(r)) / 1e3
        sk = sum(float(r["dur_us"]) for r in raw if "splitKreduce" in r["kernel"]) / 1e3
        wg = sum(float(r["dur_us"]) for r in raw if r["group"] == "weight_gemm") / 1e3
        detail = "" if GPU != "B200" else (". Weight matmuls: "
                   f"{mm:.3f} ms in the matmul kernels themselves, {sk:.3f} ms in cuBLAS split-K reductions, "
                   f"{max(wg - mm - sk, 0):.3f} ms in other kernels of the weight_gemm group (e.g. activation quantization)"
                   + (f"; {pk:.3f} ms in kernels that pack or unpack 4-bit values" if (pk := sum(
                       float(r["dur_us"]) for r in raw if "lshift" in r["kernel"]) / 1e3) else ""))
        out.append(f"**{trace}**, decode step 2: {n:,} kernels, {ms:.3f} ms of kernel time (traced){detail}\n\n" + table(
            sorted(rows, key=lambda r: -r["ms"]),
            [("group", "group", ""), ("kernel", "kernel", ""), ("grid", "blocks", ""), ("n", "count", ""),
             ("us", "µs each", ".2f"), ("ms", "total ms", ".3f")]))
    return "\n\n".join(out)


def roles(trace: str) -> dict[str, list[float]]:
    """Duration of each weight matmul by role, assigned by position (q, k, v, o, gate, up, down per layer, then lm_head),
    plus the attention kernels (cuDNN sdpa)."""
    mm = [float(r["dur_us"]) for r in step_rows(trace) if is_matmul(r)]
    assert len(mm) == LAYERS * len(ROLE_ORDER) + 1, f"{trace}: {len(mm)} matmuls"
    out = defaultdict(list)
    for i, d in enumerate(mm[:-1]):
        out[ROLE_ORDER[i % len(ROLE_ORDER)]].append(d)
    out["lm_head"].append(mm[-1])
    out["attention"] = [float(r["dur_us"]) for r in step_rows(trace) if "sdpa" in r["kernel"]]
    out["mlp"] = out["gate"] + out["up"] + out["down"]
    out["kv"] = out["k"] + out["v"]
    return out


COMPARE_TRACES = {"H100": "E3_compiled_graph", "B200": "B1_compiled_graph"}


def compare_rows() -> list[dict]:
    """H100 vs B200, same configuration (batch 1, 512-token context, CUDA graphs + fusion), kernel by kernel, with the
    time each kernel would take at its GPU's measured roof (bytes / roof). Also used by analysis/plots_blackwell.py."""
    from analysis.predict import ceilings_path, gpu_key

    traces = COMPARE_TRACES
    roof = {g: json.loads(Path(ceilings_path(gpu_key(g))).read_text())["bandwidth_Bps"] for g in traces}
    max_seq = {int(r["max_seq"]) for r in read("data/results.csv")
               if r["experiment"] in ("E3", "B1") and r["mode"] == "compiled_graph" and r["context_len"] == "512"}
    assert len(max_seq) == 1, max_seq
    attn_bytes = max_seq.pop() * KV * 2 * 2  # one layer's K and V cache (BF16), read in full by one attention call
    by = {g: roles(t) for g, t in traces.items()}
    rows = []
    for role in ["lm_head", "mlp", "gate", "up", "down", "q", "o", "kv", "k", "v", "attention"]:
        b = {**ROLE_BYTES, "mlp": ROLE_BYTES["gate"], "kv": ROLE_BYTES["k"]}.get(role, attn_bytes)
        row = {"role": {"mlp": "mlp (gate, up, down)", "kv": "kv (k, v)"}.get(role, role), "key": role, "MB": b / 1e6}
        for g in traces:
            us = statistics.mean(by[g][role])
            ideal = b / roof[g] * 1e6
            row.update({f"{g}_us": us, f"{g}_ideal": ideal, f"{g}_extra": us - ideal})
        row["speedup"] = row["H100_us"] / row["B200_us"]
        rows.append(row)
    return rows


def compare_section() -> str:
    from analysis.predict import ceilings_path, gpu_key

    traces = COMPARE_TRACES
    if not all(Path(f"data/steps/{t}_step2.csv").exists() for t in traces.values()):
        return ""
    roof = {g: json.loads(Path(ceilings_path(gpu_key(g))).read_text())["bandwidth_Bps"] for g in traces}
    rows = compare_rows()
    cols = [("role", "kernel", ""), ("MB", "bytes read (MB)", ".1f"), ("H100_us", "H100 µs", ".2f"),
            ("H100_ideal", "H100 at roof µs", ".2f"), ("H100_extra", "H100 above roof µs", ".2f"),
            ("B200_us", "B200 µs", ".2f"), ("B200_ideal", "B200 at roof µs", ".2f"),
            ("B200_extra", "B200 above roof µs", ".2f"), ("speedup", "speed-up", ".2f")]
    return ("Traces " + ", ".join(f"`{t}` ({g}, roof {roof[g] / 1e12:.3f} TB/s)" for g, t in traces.items())
            + "; decode step 2; mean µs per call. Matmul roles by position in each layer; attention = the cuDNN kernel "
            "(its second, reduction kernel not included).\n\n" + table(rows, cols))


# (NVFP4 weight-only is left out: its matmuls read BF16 copies of the weights, not the 4-bit ones.)
BW_TRACES = {"B1_compiled_graph": "bf16", "B3_fp8_B1": "fp8", "B3_nvfp4_B1": "nvfp4"}


def bandwidth_section() -> str:
    """Bytes of weights per step ÷ time in the weight-matmul kernels of one traced step (batch 1)."""
    from analysis.predict import ModelShape, ceilings_path, gpu_key, weight_bytes_per_step

    roof = json.loads(Path(ceilings_path(gpu_key(GPU))).read_text())["bandwidth_Bps"]
    gb = {"bf16": weight_bytes_per_step(ModelShape.from_config(__import__("analysis.plots", fromlist=["LLAMA"]).LLAMA)) / 1e9}
    for r in read("data/results.csv"):
        if GPU in r["gpu"] and "weight_GB=" in r["notes"]:
            gb[r["quant"]] = float(r["notes"].split("weight_GB=")[1].split()[0])
    rows = []
    for trace, q in BW_TRACES.items():
        raw = step_rows(trace)
        mm = sum(float(r["dur_us"]) for r in raw if is_matmul(r)) / 1e6
        sk = sum(float(r["dur_us"]) for r in raw if "splitKreduce" in r["kernel"]) / 1e6
        wg = sum(float(r["dur_us"]) for r in raw if r["group"] == "weight_gemm") / 1e6
        rows.append({"trace": trace, "q": q, "gb": gb[q], "a": gb[q] / mm / 1e3, "b": gb[q] / (mm + sk) / 1e3,
                     "c": gb[q] / wg / 1e3, "pct": 100 * gb[q] * 1e9 / (mm + sk) / roof})
    return (f"Roof {roof / 1e12:.3f} TB/s. Matmul kernels = cuBLAS/CUTLASS matmuls; split-K = their reduction kernels; "
            "weight_gemm group adds the kernels that quantize activations (FP8, NVFP4).\n\n" + table(rows, [
                ("trace", "trace", ""), ("q", "weights", ""), ("gb", "weight GB/step", ".2f"),
                ("a", "TB/s, matmul kernels", ".2f"), ("b", "TB/s, + split-K", ".2f"), ("pct", "% of roof (+ split-K)", ".0f"),
                ("c", "TB/s, whole weight_gemm group", ".2f")]))


def role_bandwidth_rows() -> list[dict]:
    """Each weight matmul by role (q, k, v, o, gate, up, down, lm_head) for each format at batch 1: thread blocks,
    whether cuBLAS adds a split-K reduction, mean µs per call (matmul + its reduction) and the bandwidth that gives."""
    from analysis.predict import ModelShape, weight_bytes_per_step

    gb = {"bf16": weight_bytes_per_step(ModelShape.from_config(__import__("analysis.plots", fromlist=["LLAMA"]).LLAMA)) / 1e9}
    for r in read("data/results.csv"):
        if GPU in r["gpu"] and "weight_GB=" in r["notes"]:
            gb[r["quant"]] = float(r["notes"].split("weight_GB=")[1].split()[0])
    order = [ROLE_ORDER[i % len(ROLE_ORDER)] for i in range(LAYERS * len(ROLE_ORDER))] + ["lm_head"]
    rows = []
    for trace, q in BW_TRACES.items():
        n, role = 0, None
        us, grids, splitk = defaultdict(float), defaultdict(set), defaultdict(int)
        for r in step_rows(trace):
            if is_matmul(r):
                role = order[n]
                n += 1
                grids[role].add(r["grid"])
            elif "splitKreduce" in r["kernel"]:
                splitk[role] += 1
            else:
                continue
            us[role] += float(r["dur_us"])
        assert n == len(order), f"{trace}: {n} matmuls"
        calls = Counter(order)
        for role in ROLE_ORDER + ["lm_head"]:
            b = ROLE_BYTES[role] * gb[q] / gb["bf16"]  # all linear layers, lm_head included, use the same format
            t = us[role] / calls[role]
            rows.append({"trace": trace, "q": q, "role": role, "blocks": "/".join(sorted(grids[role])),
                         "splitk": "yes" if splitk[role] else "no", "MB": b / 1e6, "us": t, "tbs": b / t / 1e6})
    return rows


def role_bandwidth_section() -> str:
    return ("One traced step per format (batch 1); mean µs per call, including the split-K reduction kernel where cuBLAS "
            "adds one. Bytes = each layer's BF16 weight bytes × (format's weight GB/step ÷ BF16's); B200 has 148 SMs.\n\n"
            + table(role_bandwidth_rows(), [
                ("q", "weights", ""), ("role", "matmul", ""), ("blocks", "thread blocks", ""), ("splitk", "split-K", ""),
                ("MB", "weight MB", ".1f"), ("us", "µs per call", ".2f"), ("tbs", "TB/s", ".2f")]))


def fixed_cost_section() -> str:
    """Latest bench/fixed_cost.py sweep on this GPU: batch-1 BF16 matmuls back to back in a CUDA graph."""
    rows = [r for r in read("data/fixed_cost.csv") if GPU in r["gpu"]] if Path("data/fixed_cost.csv").exists() else []
    if not rows:
        return ""
    sweep = max(r["sweep_id"] for r in rows)
    rows = [r for r in rows if r["sweep_id"] == sweep]
    r0 = rows[0]
    for r in rows:
        r.update({k: float(r[k]) for k in ("us_per_kernel", "TBps", "above_roof_us")})
        r["MB"] = float(r["weight_bytes"]) / 1e6
    return (f"Sweep `{sweep}`: each shape's weight copied until the copies exceed {r0['pool_gib']} GiB (≫ L2), one "
            "GEMV per copy, captured in one CUDA graph; µs per kernel = replay time ÷ kernels (median of replays). "
            f"Straight-line fit over all shapes: **fixed {float(r0['fit_fixed_us']):.2f} µs per kernel, streaming at "
            f"{float(r0['fit_Bps']) / 1e12:.2f} TB/s** (roof {float(r0['roof_Bps']) / 1e12:.3f}).\n\n"
            + table(rows, [("N", "N (out)", ""), ("K", "K (in)", ""), ("MB", "weight MB", ".1f"),
                           ("decode_role", "decode matmul", ""), ("kernels", "kernels", ""),
                           ("us_per_kernel", "µs per kernel", ".2f"), ("TBps", "TB/s", ".2f"),
                           ("above_roof_us", "above roof µs", ".2f")]))


NCU_FREE = {"B_bf16_matmuls_free": ("bf16", ["q", "k", "v", "o", "gate", "up", "down"]),
            "B_lm_head_free": ("bf16", ["lm_head"]),
            "B_fp8_matmuls_free": ("fp8", ["q", "k", "v", "o", "gate", "up", "down"]),
            "B_nvfp4_matmuls_free": ("nvfp4", ["q", "k", "v", "o", "gate", "up", "down"]),
            "B_attn_64k_free": ("bf16", ["attention, 64k context"])}


def ncu_free_section() -> str:
    """Nsight Compute with --clock-control none (data/ncu/B_*_free.csv): layer 0's matmuls in each format, lm_head,
    attention at 64k. ncu still replays each kernel alone with cold caches, so its durations aren't the step's."""
    from analysis.predict import ceilings_path, gpu_key

    roof = json.loads(Path(ceilings_path(gpu_key(GPU))).read_text())["bandwidth_Bps"]
    rows = []
    for name, (q, roles_) in NCU_FREE.items():
        path = Path(f"data/ncu/{name}.csv")
        if not path.exists():
            continue
        for role, r in zip(roles_, csv.DictReader(path.open())):
            tbs = float(r["dram_read_TBps"])
            rows.append({"q": q, "role": role, "grid": f"{float(r['grid']):.0f}", "waves": float(r["waves_per_sm"] or "nan"),
                         "tbs": tbs, "pct": 100 * tbs * 1e12 / roof, "tp": float(r["tensor_pipe_pct"] or "nan"),
                         "tpa": float(r["tensor_pipe_pct_active"] or "nan")})
    if not rows:
        return ""
    return ("Nsight Compute, `--clock-control none` (GPU clocks as in the timing runs), one decode step, layer 0. "
            f"% of roof uses the measured {roof / 1e12:.3f} TB/s. Tensor pipe: % of elapsed cycles over the whole GPU, and % "
            "of cycles while the SM is active (the SMs that have a block).\n\n" + table(rows, [
                ("q", "weights", ""), ("role", "kernel", ""), ("grid", "thread blocks", ""), ("waves", "waves per SM", ".2f"),
                ("tbs", "DRAM read TB/s", ".2f"), ("pct", "% of roof", ".0f"), ("tp", "tensor pipe % (GPU)", ".1f"),
                ("tpa", "tensor pipe % (active SMs)", ".0f")]))


def slope_section() -> str:
    """Least-squares line through the batch-1 context sweep (CUDA graphs + fusion, BF16), each GPU."""
    import numpy as np

    from analysis.predict import ModelShape, ceilings_path, gpu_key, kv_bytes_per_token

    kv = int(kv_bytes_per_token(ModelShape.from_config(__import__("analysis.plots", fromlist=["LLAMA"]).LLAMA)))
    excluded = {r["run_id"] for r in read("data/excluded_runs.csv")}
    rows = []
    for gpu, exp in (("H100", "E4"), ("B200", "B4")):
        pts = defaultdict(list)
        for r in read("data/results.csv"):
            if gpu in r["gpu"] and r["experiment"] == exp and r["run_id"] not in excluded and r["batch"] == "1":
                pts[int(r["context_len"])].append(float(r["ms_step_median"]))
        xs = sorted(pts)
        slope, icpt = np.polyfit(xs, [statistics.median(pts[x]) for x in xs], 1)
        roof = json.loads(Path(ceilings_path(gpu_key(gpu))).read_text())["bandwidth_Bps"]
        rows.append({"gpu": gpu, "pts": ", ".join(f"{x:,}" for x in xs), "icpt": icpt, "ns": slope * 1e6,
                     "pred": kv / roof * 1e9, "tbs": kv / (slope * 1e-3) / 1e12, "pct": 100 * kv / (slope * 1e-3) / roof})
    return (f"KV cache per token of context: {kv:,} bytes.\n\n" + table(rows, [
        ("gpu", "GPU", ""), ("pts", "contexts", ""), ("icpt", "intercept ms", ".2f"), ("ns", "ns per token", ".1f"),
        ("pred", "ns per token at the roof", ".1f"), ("tbs", "KV read TB/s", ".2f"), ("pct", "% of roof", ".0f")]))


def overlap_section() -> str:
    """How often a kernel starts before the previous one has finished (programmatic dependent launch)."""
    rows = []
    for gpu, traces in STEP_TRACES.items():
        for trace in traces:
            path = Path(f"data/steps/{trace}_step2.csv")
            if not path.exists():
                continue
            ks = [(float(r["start_us"]), float(r["start_us"]) + float(r["dur_us"])) for r in step_rows(trace)]
            gaps = [ks[i + 1][0] - ks[i][1] for i in range(len(ks) - 1)]
            ov = [-g for g in gaps if g < 0]
            rows.append({"trace": trace, "pairs": len(gaps), "pct": 100 * len(ov) / len(gaps),
                         "ov": statistics.median(ov) if ov else 0.0, "ov_mean": statistics.mean(ov) if ov else 0.0,
                         "gap": statistics.median([g for g in gaps if g >= 0]) if len(ov) < len(gaps) else 0.0})
    return table(rows, [("trace", "trace", ""), ("pairs", "consecutive kernel pairs", ""),
                        ("pct", "% that overlap", ".0f"), ("ov", "median overlap µs", ".2f"), ("ov_mean", "mean overlap µs", ".2f"),
                        ("gap", "median gap µs (the rest)", ".2f")])


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
            "gemv": c.get("gemv_Bps", float("nan")) / 1e12, "roof": c["bandwidth_Bps"] / 1e12, "kernel": c.get("roof_kernel", "sum"), "tf": c["peak_flops"] / 1e12,
            "ridge": c["peak_flops"] / c["bandwidth_Bps"], "date": c["date"]}],
            [("sum", "read, x.sum() TB/s", ".3f"), ("rows", "read, row-sum TB/s", ".3f"), ("copy", "copy TB/s", ".3f"), ("gemv", "GEMV TB/s", ".3f"),
             ("roof", "roof TB/s", ".3f"), ("kernel", "roof kernel", ""), ("tf", "BF16 matmul TFLOP/s", ".1f"),
             ("ridge", "ridge FLOP/B", ".0f"), ("date", "run", "")]))
    by_exp = timing_groups()
    order = ["E1", "E1-recheck", "E3", "E4", "E5", "E6", "E7", "repeat", "B1", "B2", "B3", "B4", "B5"]
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
    if GPU in STEP_TRACES:
        parts.append("## Per-kernel detail of one traced step (data/steps/, from the nsys traces; traced durations)\n\n" + steps_section())
    if GPU == "B200":
        parts.append("## H100 vs B200, kernel by kernel (batch 1, 512 context, CUDA graphs + fusion)\n\n" + compare_section())
        parts.append("## Weight-matmul bandwidth at batch 1 (one traced step)\n\n" + bandwidth_section())
        parts.append("## Weight matmuls by layer role, per format (batch 1, one traced step)\n\n" + role_bandwidth_section())
        if nf := ncu_free_section():
            parts.append("## Nsight Compute with unlocked clocks (B2 session)\n\n" + nf)
        if fc := fixed_cost_section():
            parts.append("## Fixed cost per kernel, measured directly (bench/fixed_cost.py)\n\n" + fc)
        parts.append("## Cost per token of context (straight-line fit, batch 1)\n\n" + slope_section())
        parts.append("## Overlapping kernels (a kernel starting before the previous one ends), both GPUs\n\n" + overlap_section())
    excluded = read("data/excluded_runs.csv")
    parts.append(f"## Excluded runs ({len(excluded)})\n\n" + table(excluded, [("run_id", "run ID", ""), ("reason", "reason", "")]))
    OUT.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
