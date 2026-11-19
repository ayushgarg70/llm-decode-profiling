"""Merge a CSV pulled from a GPU box into the local one, append-only.

Rows already present locally (same values in the local columns) are skipped; new rows are appended,
mapped by column name onto the LOCAL header (columns the local file doesn't have are dropped; the
per-GPU ceilings JSON keeps the full detail). Never rewrites or reorders existing rows.

    python scripts/merge_csv.py <pulled.csv> <local.csv>
"""

import csv
import sys
from pathlib import Path


def main(src: str, dst: str) -> None:
    src_rows = list(csv.DictReader(open(src)))
    if not Path(dst).exists():
        Path(dst).write_text(Path(src).read_text())
        print(f"{dst}: new file, {len(src_rows)} rows")
        return
    header = next(csv.reader(open(dst)))
    seen = {tuple(r.get(h, "") for h in header) for r in csv.DictReader(open(dst))}
    new = [r for r in src_rows if tuple(r.get(h, "") for h in header) not in seen]
    with open(dst, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore", restval="")
        w.writerows(new)
    print(f"{dst}: +{len(new)} rows ({len(src_rows) - len(new)} already present)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
