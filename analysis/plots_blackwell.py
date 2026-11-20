"""Charts for post 2 ("Does it hold on Blackwell?"): H100 vs B200, from the same CSVs as post 1.

    uv run python -m analysis.plots_blackwell            # every chart that has data
    uv run python -m analysis.plots_blackwell batch1

Each GPU is compared against its own measured roof (data/ceilings.json for H100,
data/ceilings_B200.json for B200). Outputs go to figures/b200/.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from analysis import plots as P  # shared style, palette and helpers
from analysis.predict import ModelShape, ceilings_path, gpu_key, kv_bytes_per_token, weight_bytes_per_step

plt = P.plt
FIG = Path("figures/b200")
GPU_COLOR = {"H100": "#2a78d6", "B200": "#eb6834"}
GPUS = ("H100", "B200")


def rows(gpu: str, experiments: tuple[str, ...]) -> list[dict]:
    excluded = {r["run_id"] for r in csv.DictReader(open("data/excluded_runs.csv"))}
    return [r for r in csv.DictReader(open("data/results.csv"))
            if gpu in r["gpu"] and r["experiment"] in experiments and r["run_id"] not in excluded]


def roof(gpu: str) -> dict:
    return json.loads(Path(ceilings_path(gpu_key(gpu))).read_text())


def med(rs: list[dict], **match) -> float | None:
    sel = [float(r["ms_step_median"]) for r in rs if all(r[k] == str(v) for k, v in match.items())]
    return float(np.median(sel)) if sel else None


def save(fig, name: str, caption: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, -0.02, caption, fontsize=8.5, color=P.MUTED, ha="left", va="top", wrap=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {FIG}/{name}.png/.svg")


def floor_ms(gpu: str, batch: int, context: int) -> float:
    m = ModelShape.from_config(P.LLAMA)
    return (weight_bytes_per_step(m) + batch * context * kv_bytes_per_token(m)) / roof(gpu)["bandwidth_Bps"] * 1e3


# ---------------------------------------------------------------- 1. batch 1: four launch modes on both GPUs
def batch1():
    modes = [("eager", "Eager"), ("eager_graph", "Eager +\nCUDA graph"), ("compiled", "torch.compile\n(fusion)"),
             ("compiled_graph", "Fusion +\nCUDA graph")]
    data = {"H100": rows("H100", ("E1", "E3")), "B200": rows("B200", ("B1",))}
    if not data["B200"]:
        print("batch1: no B200 data")
        return
    P.style()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(modes))
    w = 0.36
    for i, gpu in enumerate(GPUS):
        ys = [med(data[gpu], impl="own", mode=m, batch=1, context_len=512) for m, _ in modes]
        xs = x + (i - 0.5) * w
        ax.bar(xs, ys, width=w, color=GPU_COLOR[gpu], edgecolor=P.SURFACE, linewidth=2, label=gpu)
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 0.2, f"{yi:.1f}", ha="center", fontsize=8.5, color=P.INK)
        f = floor_ms(gpu, 1, 512)
        ax.axhline(f, color=GPU_COLOR[gpu], linewidth=1.0, linestyle=(0, (4, 3)))
        ax.text(len(modes) - 0.45, f + 0.15, f"{gpu} floor {f:.2f} ms", fontsize=8.5, color=P.INK2, ha="right")
    ax.set_xticks(x, [label for _, label in modes])
    ax.set_ylabel("ms per token (batch 1, 512-token context)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"{roof('B200')['bandwidth_Bps'] / roof('H100')['bandwidth_Bps']:.2f}x the memory bandwidth, not the same speed-up", pad=10)
    save(fig, "batch1", "Llama-3.1-8B BF16, our decode loop, batch 1, 512-token context. Floors = 15.0 GB of weights + the KV "
                        f"cache, ÷ each GPU's measured roof (H100 {roof('H100')['bandwidth_Bps'] / 1e12:.2f} TB/s, "
                        f"B200 {roof('B200')['bandwidth_Bps'] / 1e12:.2f} TB/s). H100 eager: median of post 1's first session "
                        "(eager moves by ~10% between sessions).")


# ---------------------------------------------------------------- 3. batch sweep on both GPUs
def batch_sweep():
    data = {"H100": rows("H100", ("E5",)), "B200": rows("B200", ("B2",))}
    if not data["B200"]:
        print("batch_sweep: no B200 data")
        return
    P.style()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for gpu in GPUS:
        bs = sorted({int(r["batch"]) for r in data[gpu]})
        ys = [b / med(data[gpu], batch=b, context_len=2048) * 1e3 for b in bs]
        ax.plot(bs, ys, color=GPU_COLOR[gpu], linewidth=2, marker="o", markersize=7, markeredgecolor=P.SURFACE, label=gpu)
        ax.text(bs[-1] * 1.08, ys[-1], f"{ys[-1]:,.0f} tok/s", va="center", fontsize=9, color=P.INK)
        v = [r for r in rows(gpu, ("E7", "B5")) if r["impl"] == "vllm" and r["mode"] == "cuda_graphs"
             and r["kv_dtype"] == "bf16" and r["context_len"] == "2048"]
        vb = sorted({int(r["batch"]) for r in v})
        if vb:
            ax.plot(vb, [b / med(v, batch=b) * 1e3 for b in vb], color=GPU_COLOR[gpu], linewidth=1, linestyle=(0, (3, 3)),
                    marker="o", markersize=7, markerfacecolor=P.SURFACE, markeredgewidth=1.5, label=f"{gpu}, vLLM")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128], ["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.set_xlim(0.8, 260)
    ax.set_xlabel("batch size (2,048-token context per request)")
    ax.set_ylabel("tokens per second (all requests)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Batching on both GPUs", pad=10)
    save(fig, "batch_sweep", "Filled: our loop (CUDA graphs + fusion, BF16, synthetic KV fill, steady-state runs). Hollow, dashed: "
                             "vLLM 0.34 with CUDA graphs, same model and context, decode time by the difference method.")


# ---------------------------------------------------------------- 4. precision on B200
def precision():
    rs = rows("B200", ("B3",))
    if not rs:
        print("precision: no B200 data")
        return
    ppl = {r["quant"]: float(r["ppl"]) for r in csv.DictReader(open("data/ppl.csv")) if "B200" in r["gpu"]} \
        if Path("data/ppl.csv").exists() else {}
    names = {"bf16": "BF16", "fp8": "FP8 (W8A8)", "nvfp4": "NVFP4 (W4A4)", "nvfp4w": "NVFP4 weights only"}
    colors = {"bf16": "#2a78d6", "fp8": "#eda100", "nvfp4": "#1baf7a", "nvfp4w": "#eb6834"}
    order = [q for q in names if any(r["quant"] == q for r in rs)]
    m = ModelShape.from_config(P.LLAMA)
    P.style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, b in zip(axes, (1, 32)):
        ys = np.arange(len(order))[::-1]
        for y, q in zip(ys, order):
            ms = med(rs, quant=q, batch=b, context_len=2048)
            if ms is None:
                ax.text(0.1, y, "did not compile", va="center", fontsize=9, color=P.INK2)
                continue
            ax.barh(y, ms, height=0.56, color=colors[q], edgecolor=P.SURFACE, linewidth=2)
            wb = next((P._weight_bytes(r, m) for r in rs if r["quant"] == q), weight_bytes_per_step(m))
            floor = (wb + b * 2048 * kv_bytes_per_token(m)) / roof("B200")["bandwidth_Bps"] * 1e3
            ax.plot([floor, floor], [y - 0.36, y + 0.36], color=P.INK, linewidth=1.4)
            ax.text(ms + 0.2, y, f"{ms:.2f} ms", va="center", fontsize=9.5, color=P.INK, fontweight="semibold")
        ax.set_yticks(ys, [f"{names[q]}\nperplexity {ppl[q]:.2f}" if q in ppl else names[q] for q in order], fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("ms per decode step (2,048-token context)")
        ax.set_title(f"Batch {b}", pad=8)
        ax.set_xlim(0, max(med(rs, quant=q, batch=b, context_len=2048) or 0 for q in order) * 1.3)
    fig.suptitle("Low precision on Blackwell", x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    save(fig, "precision", "Black tick = bandwidth floor for the bytes each format stores, at the B200's measured roof "
                           "(NVFP4 weights-only unpacks its 4-bit weights to BF16 before every matmul, so it moves far more). "
                           "torchao 0.18 (NVFP4 configs are prototypes), CUDA graphs + fusion. Perplexity: WikiText-2 test, "
                           "same windows as post 1. NVFP4 at batch 32 only compiles with TORCHINDUCTOR_NESTED_REDUCTION=0 (torch 2.15).")


# ---------------------------------------------------------------- 6. context on both GPUs
def context():
    data = {"H100": rows("H100", ("E4",)), "B200": rows("B200", ("B4",))}
    if not data["B200"]:
        print("context: no B200 data")
        return
    P.style()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for gpu in GPUS:
        pts = sorted((int(r["context_len"]), float(r["ms_step_median"])) for r in data[gpu] if r["fill"] == "synthetic")
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        slope, icpt = np.polyfit(xs, ys, 1)
        grid = np.linspace(0, 132000, 100)
        ax.plot(grid / 1000, icpt + slope * grid, color=GPU_COLOR[gpu], linewidth=1.2, alpha=0.6)
        ax.scatter(xs / 1000, ys, s=40, color=GPU_COLOR[gpu], edgecolor=P.SURFACE, linewidth=1.5, zorder=3,
                   label=f"{gpu}: +{slope * 1e6:.1f} ns per token of context")
        pred = kv_bytes_per_token(ModelShape.from_config(P.LLAMA)) / roof(gpu)["bandwidth_Bps"] * 1e9
        ax.plot(grid / 1000, icpt + pred * 1e-6 * grid, color=GPU_COLOR[gpu], linewidth=1, linestyle=(0, (3, 3)))
        v = sorted((int(r["context_len"]), float(r["ms_step_median"])) for r in rows(gpu, ("E7", "B5"))
                   if r["impl"] == "vllm" and r["mode"] == "cuda_graphs" and r["kv_dtype"] == "bf16" and r["batch"] == "1")
        if v:
            ax.scatter([c / 1000 for c, _ in v], [m for _, m in v], s=40, facecolor=P.SURFACE, edgecolor=GPU_COLOR[gpu],
                       linewidth=1.5, zorder=3, label=f"{gpu}, vLLM")
    ax.set_xlim(0, 135)
    ax.set_ylim(0, None)
    ax.set_xlabel("context length (thousands of tokens)")
    ax.set_ylabel("ms per token (batch 1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Context cost on both GPUs (dashed: slope predicted from each roof)", pad=10)
    save(fig, "context", "CUDA graphs + fusion, BF16, batch 1, synthetic KV fill. Solid lines: straight-line fits to the "
                         "measurements; dashed: the slope each GPU's measured roof predicts. Hollow: vLLM 0.34 (CUDA graphs) at the "
                         "same contexts.")


# ---------------------------------------------------------------- 5. fixed cost per kernel, both GPUs
def fixed_cost():
    """Per kernel: the time its bytes take at the GPU's measured roof, and the time above that (from data/steps)."""
    from analysis.results_table import compare_rows

    pick = [("mlp", "MLP matmuls\n(117 MB each)"), ("q", "q projection\n(34 MB)"), ("kv", "k and v projections\n(8 MB each)"),
            ("attention", "attention, 512 tokens\n(3 MB)")]
    rows = compare_rows()
    by_role = {r["key"]: r for r in rows}
    hr, br = (json.loads(Path(ceilings_path(gpu_key(g))).read_text())["bandwidth_Bps"] / 1e12 for g in GPUS)
    P.style()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    h = 0.36
    ys = np.arange(len(pick))[::-1]
    for i, gpu in enumerate(GPUS):
        for y, (role, _) in zip(ys, pick):
            r = by_role[role]
            yy = y + (0.5 - i) * h
            ideal, extra = r[f"{gpu}_ideal"], r[f"{gpu}_extra"]
            ax.barh(yy, ideal, height=h * 0.92, color=GPU_COLOR[gpu], linewidth=0)
            ax.barh(yy, extra, left=ideal, height=h * 0.92, color=GPU_COLOR[gpu], alpha=0.35, linewidth=0)
            ax.text(ideal + extra + 0.4, yy, f"{gpu}  {ideal + extra:.1f} µs", va="center", fontsize=8.5, color=P.INK)
    ax.set_yticks(ys, [label for _, label in pick], fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(by_role[k]["H100_us"] for k, _ in pick) * 1.3)
    ax.set_xlabel("µs per call (batch 1, CUDA graphs + fusion)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=P.INK2), plt.Rectangle((0, 0), 1, 1, color=P.INK2, alpha=0.35)]
    ax.legend(handles, ["moving the bytes (bytes ÷ the GPU's measured roof)", "the rest: a fixed few µs per kernel"],
              loc="lower right", fontsize=8.5)
    ax.set_title("Faster memory shrinks the bytes part, not the fixed part", pad=10)
    save(fig, "fixed_cost", "Mean time per call in one traced decode step (512-token context), H100 (blue) and B200 (orange). "
                            f"Dark: the kernel's bytes at the GPU's measured roof ({hr:.2f} and {br:.2f} TB/s); light: the rest. "
                            "MLP = gate, up and down; k and v averaged.")


# ---------------------------------------------------------------- 7. fixed cost, measured directly (B200)
def fixed_cost_sweep():
    path = Path("data/fixed_cost.csv")
    rs = [r for r in csv.DictReader(path.open()) if "B200" in r["gpu"]] if path.exists() else []
    if not rs:
        print("fixed_cost_sweep: no data")
        return
    sweep = max(r["sweep_id"] for r in rs)
    rs = sorted((r for r in rs if r["sweep_id"] == sweep and r["K"] == "4096"), key=lambda r: float(r["weight_bytes"]))
    down = [r for r in csv.DictReader(path.open()) if r["sweep_id"] == sweep and r["K"] != "4096"]
    bw = roof("B200")["bandwidth_Bps"]
    mb = np.array([float(r["weight_bytes"]) / 1e6 for r in rs])
    us = np.array([float(r["us_per_kernel"]) for r in rs])
    P.style()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    grid = np.geomspace(0.3, 1500, 200)
    ax.plot(grid, grid * 1e6 / bw * 1e6, color=P.INK2, linewidth=1, linestyle=(0, (3, 3)),
            label=f"bytes ÷ measured roof ({bw / 1e12:.2f} TB/s)")
    ax.plot(mb, us, color=GPU_COLOR["B200"], linewidth=2, marker="o", markersize=7, markeredgecolor=P.SURFACE,
            label="measured, µs per kernel")
    for r in rs + down:
        if r["decode_role"]:
            x, y = float(r["weight_bytes"]) / 1e6, float(r["us_per_kernel"])
            if r in down:
                ax.scatter([x], [y], s=40, color=GPU_COLOR["B200"], marker="s", edgecolor=P.SURFACE, zorder=3)
            ax.annotate(r["decode_role"], (x, y), textcoords="offset points", xytext=(8, -12) if r["decode_role"] == "gate, up" else (-6, 9),
                        ha="left" if r["decode_role"] == "gate, up" else "right", fontsize=8.5, color=P.INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("weight bytes read by one matmul (MB, log scale)")
    ax.set_ylabel("µs per kernel (log scale)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"A matmul that reads almost nothing still takes {us[0]:.1f} µs", pad=10)
    save(fig, "fixed_cost_sweep", "B200, batch-1 BF16 matmuls (one input row × a [N, 4096] weight; square: down, [4096, 14336]), "
                                  "hundreds captured back to back in one CUDA graph, each on its own copy of the weight so every "
                                  "call reads HBM. Labels: the shapes a decode step uses.")


CHARTS = {"batch1": batch1, "batch_sweep": batch_sweep, "precision": precision, "context": context, "fixed_cost": fixed_cost, "fixed_cost_sweep": fixed_cost_sweep}

if __name__ == "__main__":
    for name in sys.argv[1:] or CHARTS:
        CHARTS[name]()
