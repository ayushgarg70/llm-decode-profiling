"""Sample GPU clocks / power / temperature in the background during a timing run, to catch throttling.

    with Telemetry("data/telemetry/<run_id>.csv") as tel:
        ...timed work...
    tel.summary()  # medians: sm_clock_mhz, mem_clock_mhz, power_w

Does nothing (empty summary) on machines without nvidia-smi.
"""

from __future__ import annotations

import csv
import shutil
import statistics
import subprocess
from pathlib import Path

FIELDS = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu"


class Telemetry:
    def __init__(self, path: str | Path, interval_ms: int = 200):
        self.path, self.interval_ms, self.proc = Path(path), interval_ms, None

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = self.path.open("w")
            self._f.write(FIELDS + "\n")
            self._f.flush()
            self.proc = subprocess.Popen(
                ["nvidia-smi", "-i", "0", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits",
                 f"-lms={self.interval_ms}"],
                stdout=self._f, stderr=subprocess.DEVNULL,
            )
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)
            self._f.close()

    def summary(self) -> dict:
        if not self.proc or not self.path.exists():
            return {}
        rows = list(csv.DictReader(self.path.open(), skipinitialspace=True))

        def med(key):
            vals = [float(r[key]) for r in rows if r.get(key, "").replace(".", "", 1).isdigit()]
            return statistics.median(vals) if vals else ""

        return {"sm_clock_mhz": med("clocks.sm"), "mem_clock_mhz": med("clocks.mem"), "power_w": med("power.draw")}
