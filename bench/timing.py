"""Per-step timing with CUDA events (or perf_counter on CPU, for smoke tests only).

CUDA events are recorded on the GPU stream, so the time between two events is GPU-timeline time,
idle gaps included. That is the right thing to measure for decode: if the CPU is too slow to launch
kernels, the GPU waits, and that wait is part of the step.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch


def time_steps(step: Callable[[int], None], n_steps: int, device: torch.device) -> list[float]:
    """Run step(i) for i in range(n_steps); return the duration of each step in ms."""
    if device.type == "cuda":
        events = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps + 1)]
        torch.cuda.synchronize()
        for i in range(n_steps):
            events[i].record()
            step(i)
        events[-1].record()
        torch.cuda.synchronize()
        return [events[i].elapsed_time(events[i + 1]) for i in range(n_steps)]
    out = []
    for i in range(n_steps):
        t0 = time.perf_counter()
        step(i)
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def summarize(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {
        "ms_step_median": float(np.median(a)),
        "ms_step_p10": float(np.percentile(a, 10)),
        "ms_step_p90": float(np.percentile(a, 90)),
    }
