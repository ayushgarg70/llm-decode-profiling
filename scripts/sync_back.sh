#!/usr/bin/env bash
# Run on the Mac at the end of every GPU session, BEFORE terminating the instance:
# copies results, traces and profiler reports back, then lists what arrived.
# Usage: scripts/sync_back.sh <ssh-host>
set -euo pipefail
HOST=${1:?usage: scripts/sync_back.sh <ssh-host>}
cd "$(dirname "$0")/.."
# Only what the VM produces. Mac-side analysis outputs (data/kernels.csv, data/ncu/,
# data/excluded_runs.csv) must never be overwritten by the VM's copies.
mkdir -p data traces reports
# The VM's CSVs only hold what that VM produced (push.sh never sends data/), so they are MERGED into the
# Mac's copies row by row, never copied over them.
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
rsync -az --include='results.csv' --include='ceilings.csv' --include='ppl.csv' --include='fixed_cost.csv' --exclude='*' "$HOST:~/decode-profiling/data/" "$TMP/"
for f in results.csv ceilings.csv ppl.csv fixed_cost.csv; do
  [ -f "$TMP/$f" ] && python3 scripts/merge_csv.py "$TMP/$f" "data/$f"
done
rsync -az --include='ceilings.json' --include='ceilings_*.json' --include='telemetry/***' --include='env/***' \
  --exclude='*' "$HOST:~/decode-profiling/data/" data/
for d in traces reports; do
  rsync -az "$HOST:~/decode-profiling/$d/" "$d/" 2>/dev/null || echo "(no $d/ on $HOST)"
done
echo "--- synced back from $HOST:"
find data traces reports -type f -newer pyproject.toml 2>/dev/null | sort | sed 's/^/  /'
echo
echo ">>> At the END of a session: check the files above, then terminate/stop the VM and confirm in the provider console. <<<"
