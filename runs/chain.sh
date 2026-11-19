#!/usr/bin/env bash
# Runs blocks of a session run list unattended:  SESSION=s3 runs/chain.sh block ...
# Progress: logs/<session>_progress.log (one line per block start/end).
cd "$(dirname "$0")/.."
mkdir -p logs
P=logs/${SESSION:-s2}_progress.log
BLOCKS=("$@")
[ ${#BLOCKS[@]} -eq 0 ] && BLOCKS=(e4-rest e4-ncu e5 e5-ncu)
# Refuse to start if another GPU job is already running (two jobs corrupt each other's timings).
if ps -eo args | grep -qE "[d]ecode\.(run|vllm_bench)|[a]nalysis\.ppl"; then echo "$(date +%H:%M:%S) REFUSED: a GPU job is already running" >> $P; exit 1; fi
for b in "${BLOCKS[@]}"; do
  echo "$(date +%H:%M:%S) START $b" >> $P
  runs/${SESSION:-s2}.sh "$b" > "logs/${SESSION:-s2}_$b.log" 2>&1
  rc=$?   # capture before $(date) resets it
  echo "$(date +%H:%M:%S) END $b exit=$rc" >> $P
done
echo "$(date +%H:%M:%S) ALL_DONE" >> $P
