"""Group the kernels of an nsys trace into the parts of a decode step, and compute idle time.

Input: the `cuda_gpu_kern_sum` CSV that profiling/nsys_step.sh writes (`traces/<name>_cuda_gpu_kern_sum.csv`).
Kernel durations come from the GPU side and are not inflated by the profiler. Step time is NOT taken
from the trace: tracing adds CPU time to every launch (an eager step goes from 16.4 ms to ~25 ms under
nsys), so the step time comes from the matching timing run in data/results.csv, and
    idle per step = timing-run median step time - kernel time per step.

    uv run python -m analysis.group_kernels traces/E1_own_eager --steps 5 --run-id <timing run_id>
    uv run python -m analysis.group_kernels traces/E1_own_eager --steps 5 --show-mapping

Appends one row per group to data/kernels.csv (append-only).
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

# Order matters: first match wins. Patterns are matched against the full (templated) kernel name.
GROUPS = [
    ("weight_gemm", r"nvjet|gemm|gemv|cutlass|xmma|splitKreduce|_mm_|Int4|int4|int8|fp8_|scaled_mm"),
    ("attention", r"sdpa|flash|fmha|attention|attn|cudnn"),
    ("kv_cache_write", r"index_copy"),
    ("sampling", r"ArgMax|argmax"),
    ("norm_rope_elementwise", r"elementwise|reduce_kernel|Cat|Fill|index_kernel|triton_|copy"),
    ("other", r"."),
]

KERNELS_CSV = Path("data/kernels.csv")
COLUMNS = ["trace", "date", "run_id", "group", "ms_per_step", "launches_per_step", "steps_profiled", "notes"]


def classify(name: str) -> str:
    for group, pattern in GROUPS:
        if re.search(pattern, name):
            return group
    return "other"


def load(trace: str) -> list[dict]:
    path = Path(f"{trace}_cuda_gpu_kern_sum.csv")
    return list(csv.DictReader(path.open()))


def step_ms_from_results(run_id: str) -> float:
    for r in csv.DictReader(Path("data/results.csv").open()):
        if r["run_id"] == run_id:
            return float(r["ms_step_median"])
    raise KeyError(f"run_id {run_id} not in data/results.csv")


def summarize(rows: list[dict], steps: int) -> dict:
    groups = defaultdict(lambda: {"ms": 0.0, "n": 0})
    for r in rows:
        g = groups[classify(r["Name"])]
        g["ms"] += float(r["Total Time (ns)"]) / 1e6 / steps
        g["n"] += int(r["Instances"]) / steps
    return dict(groups)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="trace path without suffix, e.g. traces/E1_own_eager")
    ap.add_argument("--steps", type=int, required=True, help="decode steps captured in the trace")
    ap.add_argument("--run-id", help="timing run whose median step time gives the idle time")
    ap.add_argument("--show-mapping", action="store_true", help="print every kernel and its group")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    rows = load(args.trace)
    if args.show_mapping:
        for r in sorted(rows, key=lambda r: -float(r["Total Time (ns)"])):
            print(f'{classify(r["Name"]):>22}  {int(r["Instances"]) // args.steps:5d}/step  '
                  f'{float(r["Avg (ns)"]) / 1e3:7.1f} us  {r["Name"][:100]}')
        print()

    groups = summarize(rows, args.steps)
    kernel_ms = sum(g["ms"] for g in groups.values())
    launches = sum(g["n"] for g in groups.values())
    step_ms = step_ms_from_results(args.run_id) if args.run_id else None

    print(f"{'group':>22} {'ms/step':>8} {'launches':>9} {'share':>6}")
    total = step_ms or kernel_ms
    for name, _ in GROUPS:
        if name in groups:
            g = groups[name]
            print(f"{name:>22} {g['ms']:8.3f} {g['n']:9.0f} {100 * g['ms'] / total:5.1f}%")
    print(f"{'kernels total':>22} {kernel_ms:8.3f} {launches:9.0f} {100 * kernel_ms / total:5.1f}%")
    if step_ms:
        idle = step_ms - kernel_ms
        print(f"{'idle (gaps)':>22} {idle:8.3f} {'':>9} {100 * idle / step_ms:5.1f}%")
        print(f"{'step (timing run)':>22} {step_ms:8.3f}")
        groups["idle"] = {"ms": idle, "n": 0}

    if not args.no_log:
        new = not KERNELS_CSV.exists()
        with KERNELS_CSV.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            for name, g in groups.items():
                w.writerow({"trace": Path(args.trace).name, "date": date.today().isoformat(),
                            "run_id": args.run_id or "", "group": name, "ms_per_step": f"{g['ms']:.6g}",
                            "launches_per_step": f"{g['n']:.6g}", "steps_profiled": args.steps,
                            "notes": args.notes})
        print(f"appended {len(groups)} rows to {KERNELS_CSV}")


if __name__ == "__main__":
    main()
