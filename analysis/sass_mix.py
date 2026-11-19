"""Instruction mix from ncu SASS exports (`ncu --import r.ncu-rep --page source --print-source sass --csv`).

For each kernel: how many times each opcode family executed (warp-level instructions), focusing on
which hardware paths are used: Hopper WGMMA (HGMMA), older tensor-core MMA (HMMA), TMA copies
(UTMALDG/UTMASTG, UBLKCP), cp.async (LDGSTS), plain global loads (LDG).

    uv run python -m analysis.sass_mix reports/E2_lm_head_sass.csv reports/E4_attn_64k_sass.csv
Writes data/ncu/<name>_mix.csv.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

FAMILIES = ["HGMMA", "QGMMA", "IGMMA", "HMMA", "IMMA", "UTMALDG", "UTMASTG", "UTMAPF", "UBLKCP", "LDGSTS", "LDG", "LDS", "STG", "STS",
            "FFMA", "FMUL", "FADD", "MUFU", "SHFL", "BAR", "SYNCS"]


def opcode(src: str) -> str:
    s = src.strip()
    s = re.sub(r"^@!?U?P\w+\s+", "", s)  # predicate
    return s.split(" ")[0].split(".")[0] if s else ""


def parse(path: Path) -> list[tuple[str, Counter, Counter]]:
    kernels, name, execd, static = [], None, Counter(), Counter()
    header = None
    for row in csv.reader(path.open()):
        if row and row[0] == "Kernel Name":
            if name:
                kernels.append((name, execd, static))
            name, execd, static, header = row[1], Counter(), Counter(), None
            continue
        if row and row[0] == "Address":
            header = row
            continue
        if header and len(row) > 5:
            op = opcode(row[header.index("Source")])
            n = int(float(row[header.index("Instructions Executed")] or 0))
            execd[op] += n
            static[op] += 1
    if name:
        kernels.append((name, execd, static))
    return kernels


def main() -> None:
    Path("data/ncu").mkdir(parents=True, exist_ok=True)
    for f in sys.argv[1:]:
        path = Path(f)
        rows = []
        for name, execd, static in parse(path):
            total = sum(execd.values())
            row = {"kernel": name, "total_inst": total}
            row.update({fam: execd.get(fam, 0) for fam in FAMILIES})
            rows.append(row)
            uses = [f"{fam} {execd[fam]:,}" for fam in FAMILIES[:11] if execd.get(fam)]
            print(f"{name[:60]:60} total {total:>12,}  | " + ", ".join(uses))
        out = Path("data/ncu") / (path.stem + "_mix.csv")
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
