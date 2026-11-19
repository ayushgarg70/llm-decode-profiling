#!/usr/bin/env bash
# Per-kernel hardware counters (Nsight Compute). Run on the GPU box from ~/decode-profiling.
# Two modes:
#   step   all kernels of ONE decode step (NVTX range "decode step 0"), light sections:
#          Speed of Light + memory throughput for every kernel. Cheap enough for ~1000 kernels.
#   deep   --set full on kernels matching a regex inside an NVTX range of step 0 ("lm_head" or "layer 5").
# Usage:
#   profiling/ncu_kernel.sh step <name> <python -m args...>
#   profiling/ncu_kernel.sh deep <name> <nvtx-range|-> <kernel-regex> <python -m args...>
# Set NCU_SUDO=1 if check_gpu.sh said ncu needs sudo. ncu locks clocks to base and flushes caches
# by default, so use its *percentages*, not its durations (times come from CUDA events / nsys).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:?step|deep}; NAME=${2:?name}; shift 2
mkdir -p reports
NCU=${NCU:-$(command -v ncu || ls /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda*/bin/ncu 2>/dev/null | sort -V | tail -1)}
[ -x "$NCU" ] || { echo "ncu not found; set NCU=/path/to/ncu"; exit 1; }
PRE=(); [ "${NCU_SUDO:-0}" = 1 ] && PRE=(sudo -E)
COMMON=(--profile-from-start off --nvtx --import-source yes -f -o "reports/$NAME")
if [ "$MODE" = step ]; then
  CMD=("${PRE[@]}" "$NCU" "${COMMON[@]}" --nvtx-include "decode step 0/"
       --section SpeedOfLight --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section LaunchStats --section Occupancy
       .venv/bin/python -m "$@")
else
  RANGE=${1:?nvtx range or -}; REGEX=${2:?kernel regex}; shift 2
  NV=(); [ "$RANGE" != - ] && NV=(--nvtx-include "decode step 0/$RANGE/")
  CMD=("${PRE[@]}" "$NCU" "${COMMON[@]}" "${NV[@]}" -k "regex:$REGEX" -c "${NCU_COUNT:-3}" --set full .venv/bin/python -m "$@")
fi
echo "+ ${CMD[*]}"
"${CMD[@]}"
"${PRE[@]}" "$NCU" --import "reports/$NAME.ncu-rep" --csv --page raw > "reports/$NAME.csv" || true
ls -la reports/ | grep "$NAME"
