"""Tidy per-kernel table from an ncu raw-page CSV export (profiling/ncu_kernel.sh writes reports/<name>.csv).

For each profiled kernel: group (same mapping as analysis/group_kernels.py), NVTX range, grid/block,
duration, DRAM bytes read and achieved bandwidth (% of peak), SM throughput, tensor-pipe activity split
into Hopper WGMMA ("gmma") vs older MMA ("hmma") instructions, and how loads arrive (TMA vs
LDGSTS async copies). Missing metrics (light section sets) are left empty.

Note: ncu locks clocks to base and flushes caches by default, so its *durations* are not the real
ones; use the percentages and byte counts. Real durations come from nsys / CUDA events.

    uv run python -m analysis.ncu_summary reports/E2_lm_head reports/E2_layer5_matmuls
    uv run python -m analysis.ncu_summary reports/E2_step_all --by-group
Writes data/ncu/<name>.csv (regenerated from the report each time).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from analysis.group_kernels import classify

METRICS = {
    "duration_us": "gpu__time_duration.sum",
    "dram_read_GB": "dram__bytes_read.sum",
    "dram_read_TBps": "dram__bytes_read.sum.per_second",
    "dram_pct_peak": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",  # % of the 3.35 TB/s spec
    "dram_TBps": "dram__bytes.sum.per_second",  # read + write; present in light section sets too
    "sm_pct_peak": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_pipe_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tensor_pipe_pct_active": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",  # % of the cycles the SM was running
    "sm_issue_pct": "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
    "tensor_gmma_inst_pct": "sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_elapsed",
    "tensor_hmma_inst_pct": "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed",
    "tma_load_GB": "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "ldgsts_inst": "sm__sass_inst_executed_op_ldgsts_cache_access.sum",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
    "waves_per_sm": "launch__waves_per_multiprocessor",
}
# Units ncu may choose per report; normalise to the column's unit.
SCALE = {"duration_us": {"ns": 1e-3, "us": 1, "usecond": 1, "ms": 1e3, "msecond": 1e3},
         "dram_read_GB": {"byte": 1e-9, "Kbyte": 1e-6, "Mbyte": 1e-3, "Gbyte": 1},
         "tma_load_GB": {"byte": 1e-9, "Kbyte": 1e-6, "Mbyte": 1e-3, "Gbyte": 1},
         "dram_read_TBps": {"byte/s": 1e-12, "Kbyte/s": 1e-9, "Mbyte/s": 1e-6, "Gbyte/s": 1e-3, "Tbyte/s": 1},
         "dram_TBps": {"byte/s": 1e-12, "Kbyte/s": 1e-9, "Mbyte/s": 1e-6, "Gbyte/s": 1e-3, "Tbyte/s": 1}}
NVTX_COL_PREFIX = "thread Domain:Push/Pop_Range"


def load(report: str) -> list[dict]:
    rows = list(csv.reader(Path(f"{report}.csv").open()))
    header, units, data = rows[0], rows[1], rows[2:]
    idx = {h: i for i, h in enumerate(header)}
    nvtx_col = next((i for i, h in enumerate(header) if h.startswith(NVTX_COL_PREFIX)), None)
    out = []
    for r in data:
        k = {"kernel": r[idx["Kernel Name"]], "nvtx": r[nvtx_col] if nvtx_col is not None else ""}
        k["group"] = classify(k["kernel"])
        for col, metric in METRICS.items():
            if metric not in idx or r[idx[metric]] in ("", "n/a"):
                k[col] = ""
                continue
            v = float(r[idx[metric]].replace(",", ""))
            unit = units[idx[metric]]
            k[col] = v * SCALE.get(col, {}).get(unit, 1)
        out.append(k)
    return out


def by_group(kernels: list[dict]) -> list[dict]:
    """Duration-weighted averages per group (percentages weighted by each kernel's duration)."""
    acc = defaultdict(lambda: defaultdict(float))
    for k in kernels:
        g, d = acc[k["group"]], k["duration_us"] or 0
        g["n"] += 1
        g["duration_us"] += d
        for col in ("dram_pct_peak", "sm_pct_peak", "tensor_pipe_pct"):
            if k[col] != "":
                g[col] += k[col] * d
                g[col + "_w"] += d
        g["dram_GB"] += (k["dram_TBps"] or 0) * d * 1e-3  # TB/s x us = MB; /1e3 -> GB
    out = []
    for name, g in sorted(acc.items(), key=lambda kv: -kv[1]["duration_us"]):
        row = {"group": name, "kernels": int(g["n"]), "duration_us": g["duration_us"], "dram_GB": g["dram_GB"],
               "dram_TBps": g["dram_GB"] / g["duration_us"] * 1e3 if g["duration_us"] else ""}
        for col in ("dram_pct_peak", "sm_pct_peak", "tensor_pipe_pct"):
            row[col] = g[col] / g[col + "_w"] if g[col + "_w"] else ""
        out.append(row)
    return out


def fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+", help="report paths without suffix")
    ap.add_argument("--by-group", action="store_true")
    args = ap.parse_args()
    Path("data/ncu").mkdir(parents=True, exist_ok=True)
    for rep in args.reports:
        ks = load(rep)
        out = Path("data/ncu") / f"{Path(rep).name}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ks[0]))
            w.writeheader()
            w.writerows(ks)
        print(f"== {rep}: {len(ks)} kernels -> {out}")
        if args.by_group:
            cols = ["group", "kernels", "duration_us", "dram_GB", "dram_TBps", "dram_pct_peak", "sm_pct_peak", "tensor_pipe_pct"]
            print("  ".join(f"{c:>15}" for c in cols))
            for g in by_group(ks):
                print("  ".join(f"{fmt(g[c]):>15}" for c in cols))
        else:
            cols = ["duration_us", "dram_read_GB", "dram_read_TBps", "dram_pct_peak", "tensor_pipe_pct",
                    "tensor_gmma_inst_pct", "tensor_hmma_inst_pct", "tma_load_GB", "ldgsts_inst", "grid", "block"]
            print(f"{'kernel':42} " + " ".join(f"{c[:14]:>14}" for c in cols))
            for k in ks:
                print(f"{k['kernel'][:42]:42} " + " ".join(f"{fmt(k[c], 3)[:14]:>14}" for c in cols))


if __name__ == "__main__":
    main()
