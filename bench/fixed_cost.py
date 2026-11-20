"""Fixed cost per kernel, measured directly: batch-1 matmuls (GEMVs) of many sizes, back to back.

Decode at batch 1 is ~225 weight matmuls per step, each reading its weight once from HBM. Model each one as
    time = fixed + bytes / bandwidth
and measure it directly: for one weight shape [N, K] (BF16), capture a CUDA graph of many GEMVs, each on a
DIFFERENT copy of the weight (a pool much larger than L2, so every call reads HBM, as in decode), replay it, and
divide by the number of kernels. A least-squares line through all shapes gives the fixed cost (intercept) and
the streaming bandwidth (1 / slope). The decode shapes (k/v, q/o, gate/up, down, lm_head) are in the sweep.

Appends one row per shape to data/fixed_cost.csv (append-only; one `sweep_id` per process).

    python -m bench.fixed_cost                 # GPU: the full sweep
    python -m bench.fixed_cost --device cpu --quick --no-log   # plumbing check, no CUDA
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from datetime import date, datetime
from pathlib import Path

# (N, K): output features x input features. K=4096 sweep over N, plus the decode shapes that aren't in it.
SHAPES = [(n, 4096) for n in (64, 256, 1024, 2048, 4096, 8192, 14336, 28672, 57344, 128256)] + [(4096, 14336)]
DECODE_ROLE = {(1024, 4096): "k, v", (4096, 4096): "q, o", (14336, 4096): "gate, up", (4096, 14336): "down",
               (128256, 4096): "lm_head"}
CSV = Path("data/fixed_cost.csv")


def fit_line(bytes_: list[float], us: list[float]) -> tuple[float, float]:
    """Least squares us = a + b * bytes. Returns (fixed µs, bandwidth in bytes/s)."""
    n = len(bytes_)
    mx, my = sum(bytes_) / n, sum(us) / n
    b = sum((x - mx) * (y - my) for x, y in zip(bytes_, us)) / sum((x - mx) ** 2 for x in bytes_)
    return my - b * mx, 1e6 / b


def pool_size(nbytes: int, pool_bytes: int) -> int:
    """Copies of the weight needed so one pass through the pool reads >= pool_bytes (at least 4)."""
    return max(4, math.ceil(pool_bytes / nbytes))


def measure(n: int, k: int, device: str, pool_bytes: int, min_kernels: int, reps: int) -> dict:
    import torch

    dt = torch.bfloat16
    copies = pool_size(n * k * 2, pool_bytes)
    ws = [torch.randn(n, k, device=device, dtype=dt) for _ in range(copies)]
    x = torch.randn(1, k, device=device, dtype=dt)
    passes = max(1, math.ceil(min_kernels / copies))
    kernels = copies * passes

    def body():
        for _ in range(passes):
            for w in ws:
                torch.nn.functional.linear(x, w)  # what a decode matmul calls

    if device == "cpu":  # plumbing only
        import time

        t0 = time.perf_counter()
        body()
        times_ms = [(time.perf_counter() - t0) * 1e3]
    else:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()  # warm-up outside the graph (cuBLAS picks its kernel)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        times_ms = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            times_ms.append(a.elapsed_time(b))
        del g
    del ws
    us = statistics.median(times_ms) * 1e3 / kernels
    nbytes = n * k * 2
    return {"N": n, "K": k, "weight_bytes": nbytes, "copies": copies, "kernels": kernels, "reps": len(times_ms),
            "us_per_kernel": us, "TBps": nbytes / us / 1e6, "decode_role": DECODE_ROLE.get((n, k), "")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-gib", type=float, default=2.0, help="weight copies per shape; >> L2 (B200: 126.5 MiB)")
    ap.add_argument("--min-kernels", type=int, default=512)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="tiny shapes and pool (CPU plumbing check)")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    shapes, pool, min_k = SHAPES, int(args.pool_gib * 2**30), args.min_kernels
    if args.quick:
        shapes, pool, min_k = [(64, 256), (256, 256), (1024, 256)], 2**20, 8
    env = {}
    roof = float("nan")
    if args.device != "cpu":
        from analysis.predict import current_ceilings
        from bench.env import environment

        env = environment()
        roof = current_ceilings()[0]["bandwidth_Bps"]
    sweep_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_fixed_cost"
    rows = []
    for n, k in shapes:
        r = measure(n, k, args.device, pool, min_k, args.reps)
        r["above_roof_us"] = r["us_per_kernel"] - r["weight_bytes"] / roof * 1e6
        rows.append(r)
        print(f"N={n:6d} K={k:5d} {r['weight_bytes'] / 1e6:8.1f} MB  {r['us_per_kernel']:8.2f} µs  "
              f"{r['TBps']:5.2f} TB/s  above roof {r['above_roof_us']:6.2f} µs  {r['decode_role']}")
    a, bw = fit_line([r["weight_bytes"] for r in rows], [r["us_per_kernel"] for r in rows])
    print(f"fit: fixed {a:.2f} µs per kernel, streaming at {bw / 1e12:.2f} TB/s (roof {roof / 1e12:.2f})")
    if args.no_log:
        return
    common = {"sweep_id": sweep_id, "date": date.today().isoformat(), "gpu": env.get("gpu", ""),
              "roof_Bps": roof, "fit_fixed_us": a, "fit_Bps": bw, "pool_gib": args.pool_gib,
              "driver": env.get("driver", ""), "cuda": env.get("cuda", ""), "torch": env.get("torch", "")}
    fields = list(common) + list(rows[0])
    CSV.parent.mkdir(exist_ok=True)
    new = not CSV.exists()
    if not new:
        fields = next(csv.reader(CSV.open()))  # append-only: keep the existing header
    with CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({**common, **r})
    print(f"appended {len(rows)} rows to {CSV}")


if __name__ == "__main__":
    main()
