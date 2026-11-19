#!/usr/bin/env bash
# Timeline of a few decode steps (Nsight Systems). Run on the GPU box from ~/decode-profiling.
# Only the steps between cudaProfilerStart/Stop are captured (the script's --profile-steps K).
# Usage: profiling/nsys_step.sh <name> <python -m args...>
#   NSYS_TRACE=cuda,nvtx ...   lighter capture (no OS-runtime tracing): less profiler overhead per launch
#   profiling/nsys_step.sh E1_own_eager decode.run --mode eager --profile-steps 5 --nvtx-layers
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?name}; shift
mkdir -p traces
NSYS=${NSYS:-$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda*/bin/nsys 2>/dev/null | sort -V | tail -1)}
[ -x "$NSYS" ] || { echo "nsys not found; set NSYS=/path/to/nsys"; exit 1; }
CMD=("$NSYS" profile -o "traces/$NAME" --force-overwrite true --trace="${NSYS_TRACE:-cuda,nvtx,osrt}"
     --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node
     .venv/bin/python -m "$@")
echo "+ ${CMD[*]}"
"${CMD[@]}"
# Per-kernel summary + NVTX ranges as CSV, and an SQLite export for scripted timeline plots.
"$NSYS" stats -r cuda_gpu_kern_sum,nvtx_sum --format csv --force-export=true -o "traces/$NAME" "traces/$NAME.nsys-rep" || true
"$NSYS" export --type sqlite --force-overwrite true -o "traces/$NAME.sqlite" "traces/$NAME.nsys-rep" || true
ls -la traces/ | grep "$NAME"
