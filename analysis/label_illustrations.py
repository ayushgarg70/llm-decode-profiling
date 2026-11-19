"""Add text labels to the concept illustrations (figures/illustrations/*.png -> figures/illustrations/labeled/).

The illustrations were generated without any text (see figures/illustrations/PROMPTS.md); this script puts
the words on them so no figure has to be decoded by the reader. Coordinates are pixels of the original
1672x941 images; `pad` adds blank margin (left, top, right, bottom) for labels that don't fit inside.

    uv run python -m analysis.label_illustrations
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})

SRC = Path("figures/illustrations")
OUT = SRC / "labeled"
INK, INK2, MUTED = "#1a1a19", "#4a4a47", "#8a8a85"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FS = 15  # label font size (points at 100 dpi = ~21 px)


def draw(name: str, pad: tuple[int, int, int, int], labels: list[dict], legend: list[tuple[str, str]] | None = None):
    img = plt.imread(SRC / f"{name}.png")
    h, w = img.shape[:2]
    l, t, r, b = pad
    W, H = w + l + r, h + t + b
    bg = img[5, 5, :3]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    ax.imshow(img, extent=(l, l + w, t + h, t), zorder=0)
    for lab in labels:
        x, y = lab["xy"]
        x, y = x + l, y + t
        kw = dict(fontsize=lab.get("fs", FS), color=lab.get("color", INK), ha=lab.get("ha", "center"),
                  va=lab.get("va", "center"), fontweight=lab.get("weight", "normal"), linespacing=1.25, zorder=3)
        if "to" in lab:
            tx, ty = lab["to"]
            ax.annotate(lab["text"], xy=(tx + l, ty + t), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4, shrinkA=6, shrinkB=2, mutation_scale=14),
                        **kw)
        else:
            ax.text(x, y, lab["text"], **kw)
    if legend:
        lx, ly = 40, H - 45
        for color, text in legend:
            ax.add_patch(plt.Rectangle((lx, ly - 12), 24, 24, color=color, lw=0, zorder=3))
            t_ = ax.text(lx + 34, ly, text, fontsize=FS - 1, color=INK2, va="center", zorder=3)
            fig.canvas.draw()
            bb = t_.get_window_extent().transformed(ax.transData.inverted())
            lx = bb.x1 + 40
    ax.text(W - 30, H - 45 if not legend else H - 45, "illustration, not to scale", fontsize=FS - 3, color=MUTED,
            ha="right", va="center", style="italic", zorder=3)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=100, facecolor=bg)
    plt.close(fig)
    print(f"wrote {OUT / name}.png")


def decode_step():
    draw("decode_step_concept", (0, 20, 0, 90), [
        {"xy": (290, 905), "text": "GPU memory (HBM): the model's weights (blue)\nand the KV cache (orange)"},
        {"xy": (820, 660), "text": "memory bandwidth, the bottleneck:\nevery byte of the weights and the\nKV cache passes through here,\nonce per token", "to": (830, 515)},
        {"xy": (1305, 790), "text": "compute cores (SMs):\nmostly idle, waiting for data"},
        {"xy": (1615, 395), "text": "one new\ntoken"},
    ])


def launch_overhead():
    draw("launch_overhead_concept", (0, 0, 0, 40), [
        {"xy": (40, 150), "ha": "left", "weight": "semibold", "fs": FS + 1,
         "text": "Eager: the CPU launches every kernel separately (one arrow per launch)"},
        {"xy": (110, 525), "text": "CPU"},
        {"xy": (330, 525), "text": "GPU"},
        {"xy": (780, 528), "text": "GPU idle: it finished a kernel and waits for the next launch", "to": (572, 470),
         "ha": "left"},
        {"xy": (40, 578), "ha": "left", "weight": "semibold", "fs": FS + 1,
         "text": "CUDA graph: the whole step is recorded once, then handed over in one launch"},
        {"xy": (110, 765), "text": "CPU"},
        {"xy": (349, 740), "text": "one launch:\nthe recorded step"},
        {"xy": (580, 765), "text": "GPU"},
        {"xy": (1055, 672), "ha": "left", "text": "the same kernels, back to back:\nthe step ends much sooner"},
    ], legend=[(BLUE, "matmuls (read the weights)"), (ORANGE, "attention (reads the KV cache)"),
               (AQUA, "green outline: the recorded CUDA graph")])


def hopper_paths():
    draw("hopper_paths_concept", (330, 90, 330, 70), [
        {"xy": (450, -50), "weight": "semibold", "fs": FS + 1,
         "text": "Weights, read by the cuBLAS matmul: Hopper's newer path"},
        {"xy": (1250, -50), "weight": "semibold", "fs": FS + 1,
         "text": "KV cache, read by cuDNN attention: the older (Ampere) path"},
        {"xy": (-170, 200), "text": "4 warps work together\non one big tile\n(WGMMA, shown as\nHGMMA in the assembly)",
         "to": (200, 200)},
        {"xy": (-170, 560), "text": "the TMA copy engine\nmoves whole tiles\ninto on-chip\nshared memory\n(UTMALDG)",
         "to": (170, 600)},
        {"xy": (1835, 185), "text": "each warp issues\nits own small\nmultiply on its\nown small tile\n(HMMA)", "to": (1615, 185)},
        {"xy": (1835, 520), "text": "many threads, each\ncopying a small piece\n(LDGSTS)", "to": (1590, 520)},
        {"xy": (835, 960), "text": "GPU memory: weights (blue) and KV cache (orange)"},
    ])


def batching():
    draw("batching_concept", (0, 110, 0, 70), [
        {"xy": (250, 760), "text": "the model's weights\nin GPU memory"},
        {"xy": (548, 405), "text": "read once\nper step"},
        {"xy": (808, 725), "text": "GPU"},
        {"xy": (1150, -55), "text": "every request in the\nbatch uses the same\nweights", "to": (1068, 160)},
        {"xy": (1385, -55), "text": "each request's\nown KV cache"},
        {"xy": (1565, -55), "text": "one new token\nper request"},
    ])


if __name__ == "__main__":
    for f in (decode_step, launch_overhead, hopper_paths, batching):
        f()
