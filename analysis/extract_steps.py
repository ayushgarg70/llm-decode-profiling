"""Extract the kernels of one decode step from each nsys trace into a small CSV in data/steps/.

The nsys traces (traces/*.sqlite, *.nsys-rep) are large and stay out of git; these per-step kernel lists
are what the timeline charts and the per-kernel tables in notes/RESULTS.md are built from, so the published
repo can regenerate them.

Kernels are matched to the step by correlation ID with the CPU-side launches inside the NVTX range
"decode step N" (CUDA-graph replays run on the GPU after the CPU-side range has ended).
Columns: start_us (from the step's first kernel), dur_us, grid (blocks), block (threads), group
(analysis/group_kernels.py), kernel (demangled name, cut at 200 characters).

    uv run python -m analysis.extract_steps                      # every traces/E*.sqlite and B*.sqlite, step 2
    uv run python -m analysis.extract_steps traces/E6_fp8_B1.sqlite --step 3
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

from analysis.group_kernels import classify

OUT = Path("data/steps")
COLS = ["start_us", "dur_us", "grid", "block", "group", "kernel"]


def step_kernels(sqlite_path: str, step: int) -> list[dict]:
    db = sqlite3.connect(sqlite_path)
    rng = db.execute("SELECT start, end FROM NVTX_EVENTS WHERE COALESCE(text, (SELECT value FROM StringIds "
                     "WHERE id = textId)) = ?", (f"decode step {step}",)).fetchone()
    if rng is None:
        raise SystemExit(f"{sqlite_path}: no NVTX range 'decode step {step}'")
    rows = db.execute("""SELECT k.start, k.end, k.gridX * k.gridY * k.gridZ, k.blockX * k.blockY * k.blockZ, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
        JOIN StringIds s ON s.id = k.demangledName
        WHERE r.start >= ? AND r.start < ? ORDER BY k.start""", rng).fetchall()
    t0 = rows[0][0]
    return [{"start_us": f"{(a - t0) / 1e3:.3f}", "dur_us": f"{(b - a) / 1e3:.3f}", "grid": g, "block": blk,
             "group": classify(name), "kernel": name[:200]} for a, b, g, blk, name in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="*", help="traces/<name>.sqlite (default: every traces/E*.sqlite and B*.sqlite)")
    ap.add_argument("--step", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for t in args.traces or sorted(str(p) for g in ("E*.sqlite", "B*.sqlite") for p in Path("traces").glob(g)):
        ks = step_kernels(t, args.step)
        out = OUT / f"{Path(t).stem}_step{args.step}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(ks)
        print(f"{t}: {len(ks)} kernels -> {out}")


if __name__ == "__main__":
    main()
