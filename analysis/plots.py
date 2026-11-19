"""All charts for post 1, generated from the CSVs (never hand-typed numbers).

    uv run python -m analysis.plots            # every chart that has data
    uv run python -m analysis.plots step_breakdown context_sweep

Inputs: data/results.csv (timing runs, minus data/excluded_runs.csv), data/kernels.csv (nsys
breakdowns), data/ceilings.json (E0 roof), model shape from analysis/predict.py.
Outputs: figures/<name>.png (2x, for the post) and figures/<name>.svg. Filtered to gpu = H100.

Style: one categorical order for the parts of a decode step (validated with the dataviz skill's
palette checker: adjacent CVD dE >= 9.1, normal-vision dE >= 22.9 on the light surface; aqua and
yellow are below 3:1 contrast, so those segments always carry direct labels).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.predict import ModelShape, kv_bytes_per_token, weight_bytes_per_step  # noqa: E402

FIG = Path("figures")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
# Parts of a decode step, fixed order = fixed colours everywhere in the post.
PARTS = [
    ("weight_gemm", "Weight matmuls", "#2a78d6"),
    ("attention", "Attention", "#eb6834"),
    ("norm_rope_elementwise", "Norms, RoPE, small ops", "#1baf7a"),
    ("misc", "KV write, argmax, other", "#eda100"),
    ("idle", "GPU idle (launch gaps)", "#d9d8d2"),
]
LLAMA = {"model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336, "num_attention_heads": 32,
         "num_hidden_layers": 32, "num_key_value_heads": 8, "vocab_size": 128256, "tie_word_embeddings": False}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "text.color": INK,
    })


def save(fig, name: str, caption: str):
    FIG.mkdir(exist_ok=True)
    fig.text(0.01, -0.02, caption, fontsize=8.5, color=MUTED, ha="left", va="top", wrap=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote figures/{name}.png/.svg")


def results() -> list[dict]:
    excluded = set()
    if Path("data/excluded_runs.csv").exists():
        excluded = {r["run_id"] for r in csv.DictReader(open("data/excluded_runs.csv"))}
    return [r for r in csv.DictReader(open("data/results.csv"))
            if r["run_id"] not in excluded and "H100" in r["gpu"]]


def ceilings() -> dict:
    return json.loads(Path("data/ceilings.json").read_text())


def kernel_breakdown(trace: str) -> dict:
    """ms per step for each part, from the latest analysis of this trace in data/kernels.csv."""
    rows = [r for r in csv.DictReader(open("data/kernels.csv")) if r["trace"] == trace]
    latest = {}
    for r in rows:
        latest[r["group"]] = float(r["ms_per_step"])
    out = {k: latest.get(k, 0.0) for k in ("weight_gemm", "attention", "norm_rope_elementwise", "idle")}
    out["misc"] = sum(latest.get(k, 0.0) for k in ("kv_cache_write", "sampling", "other"))
    return out


# ---------------------------------------------------------------- chart 2: time per decode step
def step_breakdown():
    floor_ms = weight_bytes_per_step(ModelShape.from_config(LLAMA)) / ceilings()["bandwidth_Bps"] * 1e3
    bars = [("Eager\n(PyTorch)", "E1_own_eager"), ("Eager +\nCUDA graph", "E3_eager_graph"),
            ("torch.compile\n(fusion only)", "E3_compiled"), ("Fusion +\nCUDA graph", "E3_compiled_graph")]
    data = [(label, kernel_breakdown(trace)) for label, trace in bars]
    style()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(data))[::-1]
    for yi, (label, parts) in zip(y, data):
        left = 0.0
        for key, name, color in PARTS:
            w = parts[key]
            if w <= 0:
                continue
            ax.barh(yi, w, left=left, height=0.56, color=color, edgecolor=SURFACE, linewidth=2)
            if w > 0.9:  # selective direct labels on segments wide enough to hold text
                ax.text(left + w / 2, yi, f"{w:.1f}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if key in ("weight_gemm", "attention") else INK)
            left += w
        ax.text(left + 0.15, yi, f"{left:.1f} ms", va="center", fontsize=10, color=INK, fontweight="semibold")
    ax.axvline(floor_ms, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(floor_ms + 0.12, y.min() - 0.42, f"bandwidth floor {floor_ms:.2f} ms", fontsize=9, color=INK, va="center")
    ax.set_yticks(y, [d[0] for d in data])
    ax.set_xlabel("milliseconds per decode step (batch 1, 512-token context)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(sum(p.values()) for _, p in data) * 1.13)
    ax.set_title("Where one decode step's time goes: Llama-3.1-8B on one H100", pad=14)
    ax.set_ylim(y.min() - 0.65, y.max() + 0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in PARTS]
    ax.legend(handles, [n for _, n, _ in PARTS], ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.17), fontsize=9)
    fig.subplots_adjust(bottom=0.3)
    save(fig, "step_breakdown",
         "Kernel times from nsys traces; step times from separate timing runs (the profiler slows eager launches). "
         f"Floor = 15.0 GB of weights / {ceilings()['bandwidth_Bps'] / 1e12:.2f} TB/s measured.")


# ---------------------------------------------------------------- chart 4: ms per token vs context
def context_sweep():
    rows = [r for r in results() if r["experiment"] == "E4" and r["fill"] == "synthetic"]
    if not rows:
        print("context_sweep: no data")
        return
    bw = ceilings()["bandwidth_Bps"]
    m = ModelShape.from_config(LLAMA)
    w_bytes, kv = weight_bytes_per_step(m), kv_bytes_per_token(m)
    xs = np.array([int(r["max_seq"]) for r in rows]) / 1000
    ys = np.array([float(r["ms_step_median"]) for r in rows])
    slope, icpt = np.polyfit(xs * 1000, ys, 1)
    style()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    grid = np.linspace(0, 132, 200)
    ax.plot(grid, (w_bytes + grid * 1000 * kv) / bw * 1e3, color=INK2, linewidth=1.6, linestyle=(0, (4, 3)),
            label=f"predicted floor: (weights + KV bytes) / {bw / 1e12:.2f} TB/s")
    ax.plot(grid, icpt + slope * grid * 1000, color="#2a78d6", linewidth=1.2, alpha=0.5)
    ax.scatter(xs, ys, s=46, color="#2a78d6", edgecolor=SURFACE, linewidth=2, zorder=3,
               label="measured (CUDA graphs + fusion, batch 1)")
    cross = w_bytes / kv / 1000
    ax.axvline(cross, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(cross - 1.5, 1.0, f"KV bytes = weight bytes\nat {cross:.1f}k tokens",
            ha="right", va="bottom", fontsize=9, color=INK2)
    xa = 72
    ax.annotate(f"+{slope * 1e6:.1f} ns per token of context\n= KV cache read at {kv / (slope * 1e-3) / 1e12:.2f} TB/s",
                xy=(xa, icpt + slope * xa * 1000), xytext=(xa + 4, icpt + slope * xa * 1000 - 2.6),
                fontsize=9, color="#1c5cab", arrowprops=dict(arrowstyle="-", color="#86b6ef", linewidth=0.8))
    ax.set_xlim(0, 135)
    ax.set_ylim(0, None)
    ax.set_xlabel("context length (thousands of tokens already in the KV cache)")
    ax.set_ylabel("ms per token")
    ax.set_title("Each token of context adds exactly what the bandwidth predicts", pad=12)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "context_sweep",
         "Llama-3.1-8B BF16, one H100 SXM. KV cache filled synthetically (decode cost depends on bytes, not values). "
         "Steady-state runs only; see notes/LOG.md.")


# ---------------------------------------------------------------- chart 5: batch sweep
HBM_BYTES = 81559 * 2**20  # nvidia-smi memory.total on our H100 80GB HBM3


def batch_sweep():
    rows = [r for r in results() if r["experiment"] == "E5"]
    if not rows:
        print("batch_sweep: no data")
        return
    bw = ceilings()["bandwidth_Bps"]
    m = ModelShape.from_config(LLAMA)
    w_bytes, kv = weight_bytes_per_step(m), kv_bytes_per_token(m)
    max_seq = int(rows[0]["max_seq"])
    ctx = int(rows[0]["context_len"])
    bs = np.array([int(r["batch"]) for r in rows])
    ms = np.array([float(r["ms_step_median"]) for r in rows])
    grid = 2 ** np.linspace(0, 8, 200)
    pred_ms = (w_bytes + grid * max_seq * kv) / bw * 1e3
    # Upper bound on batch: all HBM minus the BF16 weights (8.03B params x 2 B) holds KV cache.
    b_fit = (HBM_BYTES - 8.03e9 * 2) / (max_seq * kv)
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256], ["1", "2", "4", "8", "16", "32", "64", "128", "256"])
        ax.axvspan(b_fit, 300, color="#f0efec", zorder=0)
        ax.set_xlim(0.85, 300)
        ax.set_xlabel("batch size (requests decoded together)")
    a1.plot(grid, grid / pred_ms * 1e3, color=INK2, linewidth=1.6, linestyle=(0, (4, 3)), label="predicted (bandwidth floor)")
    a1.scatter(bs, bs / ms * 1e3, s=46, color="#2a78d6", edgecolor=SURFACE, linewidth=2, zorder=3, label="measured")
    a1.set_ylabel("tokens per second (all requests)")
    a1.set_title("Throughput", pad=10)
    a1.legend(loc="upper left", fontsize=9)
    a1.text(b_fit * 1.05, a1.get_ylim()[1] * 0.06, f"KV cache no longer\nfits in 80 GB\n(B > {b_fit:.0f})", fontsize=8.5, color=INK2)
    a2.plot(grid, pred_ms, color=INK2, linewidth=1.6, linestyle=(0, (4, 3)))
    a2.scatter(bs, ms, s=46, color="#2a78d6", edgecolor=SURFACE, linewidth=2, zorder=3)
    a2.axhline(w_bytes / bw * 1e3, color=MUTED, linewidth=0.8)
    a2.text(20, w_bytes / bw * 1e3 - 0.4, f"weights alone ({w_bytes / bw * 1e3:.2f} ms)", fontsize=8.5, color=INK2, va="top")
    b_cross = w_bytes / (max_seq * kv)
    a2.axvline(b_cross, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    a2.text(b_cross * 0.93, a2.get_ylim()[1] * 0.9, f"KV bytes = weight bytes\n(B = {b_cross:.0f})", ha="right", fontsize=8.5, color=INK2)
    a2.set_ylabel("ms per decode step")
    a2.set_ylim(0, None)
    a2.set_title("Time per step", pad=10)
    fig.suptitle(f"Batching shares the weights: Llama-3.1-8B, {ctx:,}-token context per request", x=0.01, ha="left",
                 fontsize=13, fontweight="semibold")
    fig.tight_layout()
    save(fig, "batch_sweep",
         f"CUDA graphs + fusion, BF16, synthetic KV fill; steady-state runs. Prediction = (weights + B x KV bytes) / {bw / 1e12:.2f} TB/s. "
         "Shaded: batch sizes whose KV cache would not fit next to the weights (upper bound, ignoring workspace).")


# ---------------------------------------------------------------- chart 3: roofline (whole decode steps)
QUANT_COLORS = {"bf16": "#2a78d6", "int8": "#eb6834", "int4": "#1baf7a", "fp8": "#eda100"}
QUANT_NAMES = {"bf16": "BF16", "int8": "INT8 weights", "int4": "INT4 weights", "fp8": "FP8 weights + activations"}


def _weight_bytes(row: dict, m) -> float:
    """BF16: from the config. Quantized: the stored bytes run.py measured and wrote into notes."""
    for tok in row["notes"].split():
        if tok.startswith("weight_GB="):
            return float(tok.split("=")[1]) * 1e9
    return weight_bytes_per_step(m)


def roofline():
    from analysis.predict import attention_flops_per_token, matmul_flops_per_token

    c = ceilings()
    bw, peak = c["bandwidth_Bps"], c["peak_flops"]
    m = ModelShape.from_config(LLAMA)
    kv = kv_bytes_per_token(m)
    pts = []  # (quant, batch, intensity, achieved FLOP/s)
    for r in results():
        if r["experiment"] not in ("E5", "E6") or r["impl"] != "own" or int(r["context_len"]) != 2048:
            continue
        b, s_read = int(r["batch"]), int(r["max_seq"])  # the static cache is read in full (max_seq)
        flops = b * (matmul_flops_per_token(m) + attention_flops_per_token(m, s_read))
        byts = _weight_bytes(r, m) + b * s_read * kv
        pts.append((r["quant"], b, flops / byts, flops / (float(r["ms_step_median"]) * 1e-3)))
    if not pts:
        print("roofline: no data")
        return
    style()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    xs = np.logspace(-0.3, 3.3, 300)
    ax.plot(xs, np.minimum(xs * bw, peak) / 1e12, color=INK, linewidth=1.6)
    ridge = peak / bw
    ax.set_xlim(0.5, 2000)
    ax.set_ylim(0.5, 2000)
    p1, p2 = ax.transData.transform([(10, 10 * bw / 1e12), (100, 100 * bw / 1e12)])
    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    ax.text(70, 70 * bw / 1e12 * 1.35, f"memory roof: {bw / 1e12:.2f} TB/s", rotation=angle, rotation_mode="anchor",
            ha="center", fontsize=9, color=INK2)
    ax.text(ridge * 1.15, peak / 1e12 * 1.12, f"compute roof: {peak / 1e12:.0f} TFLOP/s (BF16)", fontsize=9, color=INK2)
    ax.axvline(ridge, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(ridge * 1.08, 0.65, f"ridge: {ridge:.0f} FLOP/B", ha="left", fontsize=8.5, color=INK2)
    s_read = max(int(r["max_seq"]) for r in results() if r["experiment"] == "E5")
    cap = (matmul_flops_per_token(m) + attention_flops_per_token(m, s_read)) / (s_read * kv)
    ax.axvline(cap, color="#86b6ef", linewidth=1.0)
    ax.text(cap * 1.07, 2.2, f"any batch size at 2k context\nstays left of {cap:.0f} FLOP/B\n(KV bytes grow with B too)",
            fontsize=8.5, color="#1c5cab")
    for q in ("bf16", "int8", "int4", "fp8"):
        sel = sorted((p for p in pts if p[0] == q), key=lambda p: p[1])
        if not sel:
            continue
        ax.plot([p[2] for p in sel], [p[3] / 1e12 for p in sel], linestyle="none", marker="o", markersize=7,
                color=QUANT_COLORS[q], markeredgecolor=SURFACE, markeredgewidth=1.5, label=QUANT_NAMES[q], zorder=3)
        if q == "bf16":
            seen = set()
            for _, b, x, y in sel:
                if b in seen or b not in (1, 8, 32, 128):
                    continue
                seen.add(b)
                ax.annotate(f"B={b}", (x, y / 1e12), xytext=(7, -11), textcoords="offset points", fontsize=8.5,
                            color=INK2, ha="left")
    ax.set_xlabel("arithmetic intensity of one decode step (FLOPs per byte read from HBM)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title("Decode never reaches the compute roof on one H100", pad=12)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "roofline",
         "Whole decode steps, 2,048-token context, CUDA graphs + fusion. FLOPs = matmuls + attention; bytes = weights "
         "(as stored) + KV cache read. Roofs measured in E0. FP8 tensor cores have a ~2x higher compute roof (not drawn).")


# ---------------------------------------------------------------- chart: E6 quantization
def _ppl() -> dict:
    rows = list(csv.DictReader(open("data/ppl.csv"))) if Path("data/ppl.csv").exists() else []
    return {r["quant"]: float(r["ppl"]) for r in rows}


def _median_ms(rows, **match) -> float | None:
    sel = [float(r["ms_step_median"]) for r in rows if all(r[k] == str(v) for k, v in match.items())]
    return float(np.median(sel)) if sel else None


def quantization():
    rows = [r for r in results() if r["experiment"] == "E6"]
    if not rows:
        print("quantization: no data")
        return
    ppl = _ppl()
    m = ModelShape.from_config(LLAMA)
    bw = ceilings()["bandwidth_Bps"]
    order = ["bf16", "int8", "int4", "fp8"]
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, b, cap in ((axes[0], 1, 8), (axes[1], 32, 26)):
        ys = np.arange(len(order))[::-1]
        for y, q in zip(ys, order):
            ms = _median_ms(rows, quant=q, batch=b, context_len=2048)
            wb = next((_weight_bytes(r, m) for r in rows if r["quant"] == q), weight_bytes_per_step(m))
            floor = (wb + b * 2048 * kv_bytes_per_token(m)) / bw * 1e3
            shown = min(ms, cap)
            ax.barh(y, shown, height=0.56, color=QUANT_COLORS[q], edgecolor=SURFACE, linewidth=2)
            ax.plot([floor, floor], [y - 0.36, y + 0.36], color=INK, linewidth=1.4)
            label = f"{ms:.1f} ms" if ms <= cap else f"{ms:,.0f} ms  (axis cut)"
            ax.text(shown + cap * 0.015, y, label, va="center", fontsize=9.5, color=INK, fontweight="semibold")
        ax.set_yticks(ys, [f"{QUANT_NAMES[q]}\nperplexity {ppl[q]:.2f}" if q in ppl else QUANT_NAMES[q] for q in order],
                      fontsize=9)
        ax.set_xlim(0, cap * 1.32)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("ms per decode step (2,048-token context)")
        ax.set_title(f"Batch {b}", pad=8)
    fig.suptitle("Fewer bytes per weight help at batch 1, but only some kernels keep it up", x=0.01, ha="left",
                 fontsize=13, fontweight="semibold")
    fig.tight_layout()
    save(fig, "quantization",
         "Black tick = bandwidth floor for the bytes each format reads. torchao 0.18 on Llama-3.1-8B, CUDA graphs + fusion. INT8/INT4 = weight-only (INT4: groups of 128, tinygemm); "
         "FP8 = weights + activations (cuBLAS FP8). Perplexity: WikiText-2 test, 289k tokens (BF16 6.31).")


# ---------------------------------------------------------------- chart: E7 vLLM reference + FP8 KV cache
def vllm_reference():
    rows = results()
    v = [r for r in rows if r["experiment"] == "E7"]
    if not v:
        print("vllm_reference: no data")
        return
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    bs = [1, 8, 32]
    series = [
        ("Our loop (graphs + fusion)", "#2a78d6",
         [_median_ms([r for r in rows if r["experiment"] == "E5"], batch=b, context_len=2048) for b in bs]),
        ("vLLM (CUDA graphs)", "#eb6834", [_median_ms(v, mode="cuda_graphs", kv_dtype="bf16", batch=b, context_len=2048) for b in bs]),
        ("vLLM --enforce-eager", "#1baf7a", [_median_ms(v, mode="eager", kv_dtype="bf16", batch=b, context_len=2048) for b in bs]),
    ]
    x = np.arange(len(bs))
    w = 0.26
    for i, (name, color, ys) in enumerate(series):
        xs = x + (i - 1) * w
        a1.bar(xs, ys, width=w, color=color, edgecolor=SURFACE, linewidth=2, label=name)
        for xi, yi in zip(xs, ys):
            a1.text(xi, yi + 0.2, f"{yi:.1f}", ha="center", fontsize=8.5, color=INK)
    a1.set_xticks(x, [f"B = {b}" for b in bs])
    a1.set_ylabel("ms per decode step (2,048-token context)")
    a1.grid(axis="x", visible=False)
    a1.legend(loc="upper left", fontsize=8.5, ncol=1)
    a1.set_ylim(0, 19)
    a1.set_title("Same model, same GPU", pad=8)
    for kv, color, name in (("bf16", "#2a78d6", "BF16 KV cache"), ("fp8", "#eb6834", "FP8 KV cache")):
        pts = sorted((int(r["context_len"]), float(r["ms_step_median"])) for r in v
                     if r["kv_dtype"] == kv and r["batch"] == "1" and r["mode"] == "cuda_graphs" and int(r["context_len"]) > 2048)
        xs_, ys_ = np.array([p[0] for p in pts]) / 1000, np.array([p[1] for p in pts])
        a2.plot(xs_, ys_, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, label=name)
        slope = (ys_[-1] - ys_[0]) / ((xs_[-1] - xs_[0]) * 1000) * 1e6
        a2.text(xs_[-1] - 2, ys_[-1] + 0.35, f"+{slope:.1f} ns per token", ha="right", fontsize=9, color=INK2)
    a2.set_xlabel("context length (thousands of tokens)")
    a2.set_ylabel("ms per decode step (batch 1)")
    a2.set_ylim(0, 14)
    a2.set_xlim(0, 140)
    a2.legend(loc="upper left", fontsize=8.5)
    a2.set_title("vLLM: an FP8 KV cache halves the cost of context", pad=8)
    fig.suptitle("vLLM as a reference point", x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    save(fig, "vllm_reference",
         "vLLM 0.34.0 (offline API), decode time by the difference method (T(257 tokens) - T(1 token)) / 256, median of 5; "
         "prefix caching off. Our loop: E5 runs (static cache sized 2,326 tokens).")


# ---------------------------------------------------------------- chart 1: GPU timelines from the nsys traces
GROUP_COLOR = {"weight_gemm": "#2a78d6", "attention": "#eb6834", "norm_rope_elementwise": "#1baf7a",
               "kv_cache_write": "#eda100", "sampling": "#eda100", "other": "#eda100"}


def _step_kernels(trace: str, step: int = 2) -> list[tuple[float, float, str]]:
    """Kernels (start_ms, end_ms, group) of one decode step, from data/steps/ (written from the nsys trace by
    analysis/extract_steps.py, so the chart can be regenerated without the large trace files)."""
    rows = csv.DictReader(open(f"data/steps/{trace}_step{step}.csv"))
    return [(float(r["start_us"]) / 1e3, (float(r["start_us"]) + float(r["dur_us"])) / 1e3, r["group"]) for r in rows]


def _timeline(modes, name: str, suptitle: str, caption: str, height: float, annotate_gap: bool = False):
    """GPU timelines of one decode step per mode: the full step, then a 0.4 ms zoom."""
    data = [(label, _step_kernels(tr)) for label, tr in modes]
    style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, height), gridspec_kw={"height_ratios": [1.25, 1]})
    window = (2.0, 2.4)
    for ax, win, title in ((a1, None, "One decode step on the GPU (each sliver is a kernel; background = GPU idle)"),
                           (a2, window, "Zoom: 0.4 ms of the same step" + ("s" if len(data) > 1 else ""))):
        ys = np.arange(len(data))[::-1]
        for y, (label, ks) in zip(ys, data):
            span = ks[-1][1]
            lo, hi = win or (0, span)
            ax.add_patch(plt.Rectangle((lo, y - 0.3), min(hi, span) - lo, 0.6, color="#ebeae5", lw=0))
            for group in GROUP_COLOR:
                segs = [(a, b - a) for a, b, g in ks if g == group and b > lo and a < hi]
                if segs:
                    ax.broken_barh(segs, (y - 0.3, 0.6), color=GROUP_COLOR[group], lw=0)
            if win is None:
                busy = sum(b - a for a, b, _ in ks)
                ax.text(span + 0.25, y, f"{span:.1f} ms · {len(ks):,} kernels · GPU busy {100 * busy / span:.0f}%",
                        va="center", fontsize=9, color=INK)
            elif annotate_gap:
                inside = [(a, b) for a, b, _ in ks if b > lo and a < hi]
                gap = max(((inside[i][1], inside[i + 1][0]) for i in range(len(inside) - 1)), key=lambda g: g[1] - g[0])
                mid = (gap[0] + gap[1]) / 2
                ax.annotate("GPU idle: waiting for the CPU\nto launch the next kernel",
                            xy=(mid, y + 0.3), xytext=(mid, y + 0.75), ha="center", va="bottom", fontsize=9, color=INK,
                            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
        ax.set_yticks(ys, [d[0] for d in data], fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(data) - 0.4 + (0.9 if (win and annotate_gap) else 0))
        ax.set_title(title, fontsize=11, pad=6)
        if win:
            ax.set_xlim(*win)
            ax.set_xlabel("milliseconds since the step's first kernel")
        else:
            ax.set_xlim(0, max(d[1][-1][1] for d in data) * 1.45)
    names = {"weight_gemm": "Weight matmuls", "attention": "Attention", "norm_rope_elementwise": "Norms, RoPE, small ops",
             "other": "KV write, argmax, other"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in names] + [plt.Rectangle((0, 0), 1, 1, color="#ebeae5")]
    a1.legend(handles, list(names.values()) + ["GPU idle"], ncol=5, loc="upper left", fontsize=8.5,
              bbox_to_anchor=(0, 1.0))
    a1.set_ylim(-0.6, len(data) + 0.5)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    save(fig, name, caption)


def eager_timeline():
    """E1: the eager step alone (the four-way comparison is E3's chart, `timeline`)."""
    _timeline([("Eager PyTorch\n(our loop)", "E1_own_eager_light")], "eager_timeline",
              "Eager decode: the GPU spends much of each step waiting (Llama-3.1-8B, batch 1, 512-token context)",
              "Kernels of decode step 2 from an nsys trace. Tracing adds CPU time to every launch, so this traced step "
              "is longer than in the clean timing runs (16.4 ms); kernel durations are not affected.",
              height=4.6, annotate_gap=True)


def timeline():
    modes = [("Eager PyTorch", "E1_own_eager_light"), ("Eager + CUDA graph", "E3_eager_graph"),
             ("torch.compile (fusion)", "E3_compiled"), ("Fusion + CUDA graph", "E3_compiled_graph")]
    _timeline(modes, "timeline", "The same work, launched four ways: Llama-3.1-8B, batch 1, 512-token context",
              "Kernels of decode step 2 from nsys traces (--cuda-graph-trace=node). Tracing slows the CPU side, so the two modes "
              "without CUDA graphs look longer than in clean timing runs (eager 16.4 ms, fusion only 7.05 ms); graph modes match (8.38, 6.63 ms).",
              height=6.4)


CHARTS = {"eager_timeline": eager_timeline, "timeline": timeline, "step_breakdown": step_breakdown, "context_sweep": context_sweep, "batch_sweep": batch_sweep, "roofline": roofline, "quantization": quantization, "vllm_reference": vllm_reference}

if __name__ == "__main__":
    for name in sys.argv[1:] or CHARTS:
        CHARTS[name]()
