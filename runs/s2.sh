#!/usr/bin/env bash
# Session S2 run list: E3 (CUDA graphs vs fusion), E4 (context sweep), E5 (batch sweep).
# Run block by block:  runs/s2.sh e3 | e3-profile | e4 | e4-rest | e4-ncu | e5 | e5-ncu
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R="$PY -m decode.run"
# Steady state only: the first process that compiles a new shape runs ~0.28 ms/step faster (smaller
# in-graph kernel gaps) than every later process. So each config gets one throwaway warm-up process.
warm() { $R "$@" --no-log --steps 32 --warmup 10 >/dev/null 2>&1 || true; }

case "${1:?block}" in
e3)          # the 2x2: {eager, fused} x {no graph, graph}; B=1, s=512, 3 repeats each
  for mode in eager eager_graph compiled compiled_graph; do
    for i in 1 2 3; do $R --experiment E3 --mode $mode --batch 1 --context 512; done
  done ;;
e3-profile)  # timelines of 5 steps for the three non-eager modes (eager is E1_own_eager)
  for mode in eager_graph compiled compiled_graph; do
    profiling/nsys_step.sh E3_$mode decode.run --mode $mode --context 512 --profile-steps 5
  done ;;
e4)          # context sweep, B=1, CUDA graphs + fusion; synthetic KV fill (same bytes as real)
  for s in 512 2048 8192 16384 32768 65536 130794; do   # 130794 + 278 = 131072 = max positions
    warm --mode compiled_graph --batch 1 --context $s --fill synthetic
    for i in 1 2; do $R --experiment E4 --mode compiled_graph --batch 1 --context $s --fill synthetic; done
  done
  # cross-check: real prefill vs synthetic fill at 8k must give the same step time
  $R --experiment E4 --mode compiled_graph --batch 1 --context 8192 --fill prefill --notes "real-prefill cross-check" ;;
e4-rest)     # completes the first e4 pass: ~128k point and the real-prefill cross-check (warm runs)
  warm --mode compiled_graph --batch 1 --context 130794 --fill synthetic
  for i in 1 2; do $R --experiment E4 --mode compiled_graph --batch 1 --context 130794 --fill synthetic; done
  $R --experiment E4 --mode compiled_graph --batch 1 --context 8192 --fill prefill --notes "real-prefill cross-check" ;;
e4-128k)     # one more steady-state repeat at ~128k
  warm --mode compiled_graph --batch 1 --context 130794 --fill synthetic
  $R --experiment E4 --mode compiled_graph --batch 1 --context 130794 --fill synthetic ;;
e4-ncu)      # the attention kernel at 64k context: bandwidth, grid size vs 132 SMs
  NCU_COUNT=1 profiling/ncu_kernel.sh deep E4_attn_64k "layer 5" 'sdpa|flash|fmha|attn|cudnn_generated' \
    decode.run --mode eager --context 65536 --fill synthetic --profile-steps 1 --nvtx-layers
  profiling/nsys_step.sh E4_compiled_graph_64k decode.run --mode compiled_graph --context 65536 --fill synthetic --profile-steps 5 ;;
e5)          # batch sweep at s=2048, CUDA graphs + fusion, synthetic fill
  for b in 1 2 4 8 16 32 64 128; do
    warm --mode compiled_graph --batch $b --context 2048 --fill synthetic
    for i in 1 2; do $R --experiment E5 --mode compiled_graph --batch $b --context 2048 --fill synthetic; done
  done ;;
e5-ncu)      # the same layer-5 matmuls at B=1, 8, 64: DRAM % vs tensor-core %
  MM='gemm|gemv|nvjet|xmma|cutlass'
  for b in 1 8 64; do
    NCU_COUNT=7 profiling/ncu_kernel.sh deep E5_layer5_matmuls_B$b "layer 5" "$MM" \
      decode.run --mode eager --batch $b --context 2048 --fill synthetic --profile-steps 1 --nvtx-layers
  done ;;
*) echo "unknown block"; exit 1 ;;
esac
