"""E0: measure this GPU's real ceilings, the roof for every later comparison.

  read bandwidth   two reductions over a 4 GiB BF16 tensor: x.sum(), and x.view(8192, -1).sum(1)
                   (many independent rows); bytes = tensor size (each byte read once)
  copy bandwidth   y.copy_(x); bytes = 2 x tensor size (read x + write y)
  roof             the best of these streaming kernels. One kernel alone isn't a reliable roof: on the H100,
                   x.sum() reached 3.08 TB/s while the row-sum reached 3.15 TB/s.
  BF16 matmul      8192 x 8192 x 8192; FLOPs = 2 * M * N * K

The tensor is ~80x larger than the H100's 50 MB L2 cache, so we measure HBM, not cache.
Writes data/ceilings.json on H100, data/ceilings_<GPU>.json elsewhere (analysis/predict.py ceilings_path) and appends to data/ceilings.csv.
Every variant is recorded; `bandwidth_Bps` (the roof used everywhere) is the best of them.

    python -m bench.ceilings
"""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch

from analysis.predict import DATASHEET, ceilings_path, gpu_key
from bench.env import environment
from bench.telemetry import Telemetry


def bench(fn, reps: int, warmup: int = 5) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return out


def main(gib: int = 4, reps: int = 30, n: int = 8192) -> None:
    assert torch.cuda.is_available(), "E0 needs a GPU"
    numel = gib * 1024**3 // 2
    x = torch.randn(numel, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    nbytes = x.numel() * x.element_size()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    with Telemetry(f"data/telemetry/{stamp}_E0.csv") as tel:
        t_read = bench(lambda: x.sum(), reps)
        t_rows = bench(lambda: x.view(8192, -1).sum(1), reps)
        t_copy = bench(lambda: y.copy_(x), reps)
        t_mm = bench(lambda: torch.mm(a, b), reps)

    read = nbytes / (statistics.median(t_read) / 1e3)
    rows = nbytes / (statistics.median(t_rows) / 1e3)
    copy = 2 * nbytes / (statistics.median(t_copy) / 1e3)
    flops = 2 * n**3 / (statistics.median(t_mm) / 1e3)
    env = environment()
    result = {
        "date": stamp, "gpu": env["gpu"], "gpu_form_factor": env["gpu_form_factor"],
        "bandwidth_Bps": max(read, rows, copy), "read_Bps": read, "read_rows_Bps": rows, "copy_Bps": copy,
        "roof_kernel": max((read, "sum"), (rows, "row-sum"), (copy, "copy"))[1], "peak_flops": flops,
        "read_ms_median": statistics.median(t_read), "copy_ms_median": statistics.median(t_copy),
        "matmul_ms_median": statistics.median(t_mm),
        "read_Bps_best": nbytes / (min(t_read) / 1e3), "matmul_flops_best": 2 * n**3 / (min(t_mm) / 1e3),
        "tensor_GiB": gib, "reps": reps, "matmul_n": n, **tel.summary(),
        "driver": env["driver"], "cuda": env["cuda"], "torch": env["torch"],
    }
    Path("data").mkdir(exist_ok=True)
    Path(ceilings_path(gpu_key(env["gpu"]))).write_text(json.dumps(result, indent=1))
    log = Path("data/ceilings.csv")
    new = not log.exists()
    fields = list(result) if new else next(csv.reader(log.open()))  # append-only: keep the existing header
    with log.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(result)

    ds = DATASHEET[gpu_key(env["gpu"])]
    for name, v in (("read (sum)", read), ("read (row-sum)", rows), ("copy", copy)):
        print(f"{name:15} {v / 1e12:.3f} TB/s  ({100 * v / ds['bandwidth_Bps']:.1f}% of {ds['bandwidth_Bps'] / 1e12:.2f} TB/s datasheet)")
    roof = result["bandwidth_Bps"]
    print(f"roof = {result['roof_kernel']}: {roof / 1e12:.3f} TB/s")
    print(f"BF16 matmul {flops / 1e12:.1f} TFLOPS ({100 * flops / ds['peak_flops']:.1f}% of {ds['peak_flops'] / 1e12:.0f} datasheet)")
    print(f"ridge (measured) {flops / roof:.0f} FLOPs/byte | clocks/power: {tel.summary()}")


if __name__ == "__main__":
    main()
