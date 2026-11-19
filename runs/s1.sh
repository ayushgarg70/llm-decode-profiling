#!/usr/bin/env bash
# Session S1 run list (GPU box, from ~/decode-profiling): 8B parity, E0, E1, E2.
# Run block by block:  runs/s1.sh parity | e0 | e1 | e1-profile | e2-ncu
# Timing runs and profiling runs are separate; nothing here profiles while timing.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CTX=512

case "${1:?block}" in
parity)      # our loop vs HF on the real model; paste the output into notes/LOG.md
  $PY -m decode.parity_8b --context $CTX --new-tokens 64 ;;
e0)          # ceilings, 3 repeats to check they agree
  for i in 1 2 3; do $PY -m bench.ceilings; done ;;
e1)          # eager baselines, batch 1, 512-token prompt, 256 timed steps
  for i in 1 2; do $PY -m decode.hf_baseline --experiment E1 --batch 1 --context $CTX; done
  for i in 1 2 3; do $PY -m decode.run --experiment E1 --mode eager --batch 1 --context $CTX; done ;;
e1-profile)  # nsys timelines of 5 decode steps
  profiling/nsys_step.sh E1_hf_loop decode.hf_baseline --only loop --context $CTX --profile-steps 5
  profiling/nsys_step.sh E1_own_eager decode.run --mode eager --context $CTX --profile-steps 5
  profiling/nsys_step.sh E1_own_eager_layers decode.run --mode eager --context $CTX --profile-steps 5 --nvtx-layers ;;
e2-ncu)      # hardware counters: every kernel of one step (light), then deep dives
  profiling/ncu_kernel.sh step E2_step_all decode.run --mode eager --context $CTX --profile-steps 1 --nvtx-layers
  # matmul kernel names vary by cuBLAS version (gemv / gemm / nvjet / xmma / cutlass): check nsys first
  MM='gemm|gemv|nvjet|xmma|cutlass'
  NCU_COUNT=1 profiling/ncu_kernel.sh deep E2_lm_head lm_head "$MM" decode.run --mode eager --context $CTX --profile-steps 1 --nvtx-layers
  NCU_COUNT=7 profiling/ncu_kernel.sh deep E2_layer5_matmuls "layer 5" "$MM" decode.run --mode eager --context $CTX --profile-steps 1 --nvtx-layers ;;
*) echo "unknown block"; exit 1 ;;
esac
