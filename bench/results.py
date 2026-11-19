"""Append-only results table: data/results.csv (schema below).

Never hand-edit the CSV. Every row is written by a script, once.
Added to §6: `mode` (eager | eager_graph | compiled | compiled_graph), `max_seq` (static cache size:
masked attention reads all of it), `fill` (prefill | synthetic KV), `warmup`, and the median
SM clock / memory clock / power during the run (from bench/telemetry.py).
The kernel-breakdown columns stay empty here; nsys breakdowns go to data/kernels.csv, joined by run_id.
"""

from __future__ import annotations

import csv
import uuid
from datetime import datetime
from pathlib import Path

RESULTS = Path("data/results.csv")

COLUMNS = [
    "run_id", "date", "experiment", "impl", "mode", "quant", "kv_dtype", "batch", "context_len",
    "max_seq", "fill", "new_tokens", "warmup",
    "ms_step_median", "ms_step_p10", "ms_step_p90", "tok_per_s", "pred_ms_floor", "pct_of_floor",
    "kernels_per_step", "idle_ms_per_step", "weight_gemm_ms", "attention_ms", "other_ms", "ppl",
    "sm_clock_mhz", "mem_clock_mhz", "power_w",
    "gpu", "gpu_form_factor", "driver", "cuda", "torch", "transformers", "torchao", "notes",
]


def new_run_id(experiment: str, impl: str, mode: str, batch: int, context: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{experiment}_{impl}-{mode}_B{batch}_s{context}_{uuid.uuid4().hex[:4]}"


def append_row(row: dict, path: Path = RESULTS) -> None:
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"not in the results schema: {sorted(unknown)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: _fmt(row.get(k, "")) for k in COLUMNS})


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return v
