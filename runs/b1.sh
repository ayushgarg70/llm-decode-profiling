#!/usr/bin/env bash
# Blackwell session (post 2), 1x B200. Mirrors S1-S3 with the debugged code; experiment IDs B0-B4.
#   runs/b1.sh b0 | b1 | b1-profile | b2 | b3 | b3-ppl | b3-profile | ncu | sass | b4
# Unattended:  SESSION=b1 runs/chain.sh b0 b1 b2 b3 b4 b1-profile b3-profile ncu b3-ppl sass
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R="$PY -m decode.run"
warm() { $R "$@" --no-log >/dev/null 2>&1 || true; }   # identical args: compiles the same shape (steady state)

case "${1:?block}" in
b0)          # ceilings -> data/ceilings_B200.json (H100's data/ceilings.json is untouched)
  $PY -m bench.ceilings ;;
b1)          # batch 1, s=512: eager vs CUDA graphs vs fusion (mirrors E1 + E3), 3 runs each
  for mode in eager eager_graph compiled compiled_graph; do
    warm --mode $mode --batch 1 --context 512
    for i in 1 2 3; do $R --experiment B1 --mode $mode --batch 1 --context 512; done
  done ;;
b1-profile)  # nsys timelines, 5 steps each (eager also with per-layer NVTX for the kernel map)
  profiling/nsys_step.sh B1_own_eager decode.run --mode eager --context 512 --profile-steps 5
  for mode in eager_graph compiled compiled_graph; do
    profiling/nsys_step.sh B1_$mode decode.run --mode $mode --context 512 --profile-steps 5
  done ;;
b2)          # batch sweep at s=2048 (mirrors E5)
  for b in 1 8 32 128; do
    warm --mode compiled_graph --batch $b --context 2048 --fill synthetic
    for i in 1 2; do $R --experiment B2 --mode compiled_graph --batch $b --context 2048 --fill synthetic; done
  done ;;
b3)          # precision: BF16 / FP8 / NVFP4 (W4A4) / NVFP4 weight-only at B=1 and B=32, s=2048 (mirrors E6)
  for q in bf16 fp8 nvfp4 nvfp4w; do
    for b in 1 32; do
      warm --mode compiled_graph --quant $q --batch $b --context 2048 --fill synthetic
      for i in 1 2; do $R --experiment B3 --mode compiled_graph --quant $q --batch $b --context 2048 --fill synthetic || echo "FAILED B3 $q B$b"; done
    done
  done ;;
b3-retry)    # NVFP4 at B=32 failed to compile (Inductor "Cycle detected in memory planning graph"); retry without the
             # peak-memory reorder pass, plus BF16 B=32 with the same flag as a control
  for q in nvfp4 bf16; do
    warm --mode compiled_graph --quant $q --batch 32 --context 2048 --fill synthetic --no-peak-memory-reorder
    for i in 1 2; do $R --experiment B3 --mode compiled_graph --quant $q --batch 32 --context 2048 --fill synthetic --no-peak-memory-reorder || echo "FAILED B3-retry $q B32"; done
  done ;;
b3-retry2)   # retry 1 failed one pass later at the same fusion (KeyError: FusedNestedReductions); turn that fusion off
  export TORCHINDUCTOR_NESTED_REDUCTION=0
  for q in nvfp4 bf16; do
    warm --mode compiled_graph --quant $q --batch 32 --context 2048 --fill synthetic --notes "TORCHINDUCTOR_NESTED_REDUCTION=0"
    for i in 1 2; do $R --experiment B3 --mode compiled_graph --quant $q --batch 32 --context 2048 --fill synthetic --notes "TORCHINDUCTOR_NESTED_REDUCTION=0" || echo "FAILED B3-retry2 $q B32"; done
  done ;;
b3-ppl)      # same WikiText-2 windows as post 1
  for q in bf16 fp8 nvfp4 nvfp4w; do $PY -m analysis.ppl --quant $q --notes "B200" || echo "FAILED ppl $q"; done ;;
b3-profile)  # which kernels each format runs, B=1 and B=32
  for q in fp8 nvfp4 nvfp4w; do for b in 1 32; do
    profiling/nsys_step.sh B3_${q}_B$b decode.run --mode compiled_graph --quant $q --batch $b --context 2048 --fill synthetic --profile-steps 3 || echo "FAILED nsys $q B$b"
  done; done ;;
ncu)         # hardware counters: lm_head + layer-5 matmuls (eager, NVTX), NVFP4 kernels, attention at 64k
  NCU_COUNT=1 profiling/ncu_kernel.sh deep B_lm_head lm_head '.' decode.run --mode eager --context 512 --profile-steps 1 --nvtx-layers
  NCU_COUNT=7 profiling/ncu_kernel.sh deep B_layer5_matmuls "layer 5" 'gemm|nvjet|cutlass|sm100' decode.run --mode eager --context 512 --profile-steps 1 --nvtx-layers
  NCU_COUNT=12 profiling/ncu_kernel.sh deep B_nvfp4_kernels - '.' decode.run --mode compiled_graph --quant nvfp4 --context 2048 --fill synthetic --profile-steps 1
  NCU_COUNT=1 profiling/ncu_kernel.sh deep B_attn_64k "layer 5" 'sdpa|flash|fmha|attn|cudnn_generated' decode.run --mode eager --context 65536 --fill synthetic --profile-steps 1 --nvtx-layers ;;
sass)        # instruction listings (only the VM's ncu can export them)
  NCU=$(ls /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda*/bin/ncu 2>/dev/null | sort -V | tail -1)
  for r in B_lm_head B_layer5_matmuls B_nvfp4_kernels B_attn_64k; do
    [ -f reports/$r.ncu-rep ] && $NCU --import reports/$r.ncu-rep --page source --print-source sass --csv > reports/${r}_sass.csv || true
  done ;;
b4)          # context: 2k, 32k, ~128k at B=1 (mirrors E4)
  for s in 2048 32768 130794; do
    warm --mode compiled_graph --batch 1 --context $s --fill synthetic
    for i in 1 2; do $R --experiment B4 --mode compiled_graph --batch 1 --context $s --fill synthetic; done
  done ;;
*) echo "unknown block $1"; exit 1 ;;
esac
