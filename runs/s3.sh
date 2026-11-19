#!/usr/bin/env bash
# Session S3 run list: E6 (quantization: speed + perplexity) and E7 (vLLM reference, FP8 KV cache).
#   runs/s3.sh e6 | e6-ppl | e6-profile | e6-ncu | e7 | repeat
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R="$PY -m decode.run"
V=".venv-vllm/bin/python -m decode.vllm_bench"
# Steady state only: one throwaway process with IDENTICAL arguments compiles the shape first (see S2 LOG).
warm() { $R "$@" --no-log >/dev/null 2>&1 || true; }

case "${1:?block}" in
e6)          # 4 weight formats x {B=1 s=2k, B=32 s=2k, B=1 s=32k}; CUDA graphs + fusion; synthetic KV fill
  for q in bf16 int8 int4 fp8; do
    for cfg in "1 2048" "32 2048" "1 32768"; do
      set -- $cfg
      warm --mode compiled_graph --quant $q --batch $1 --context $2 --fill synthetic
      for i in 1 2; do $R --experiment E6 --mode compiled_graph --quant $q --batch $1 --context $2 --fill synthetic; done
    done
  done ;;
e6-ppl)      # quality: WikiText-2 test perplexity, same windows for every format
  for q in bf16 int8 int4 fp8; do $PY -m analysis.ppl --quant $q; done ;;
e6-profile)  # nsys: 5 steps per format at B=1 s=2k (kernel names + time per group)
  for q in int8 int4 fp8; do
    profiling/nsys_step.sh E6_${q}_B1 decode.run --mode compiled_graph --quant $q --context 2048 --fill synthetic --profile-steps 5
  done ;;
e6-ncu)      # ncu --set full on the first matmul kernels of one step (graph replay, no NVTX layer ranges)
  NCU_COUNT=12 profiling/ncu_kernel.sh deep E6_int4_kernels - '.' \
    decode.run --mode compiled_graph --quant int4 --context 2048 --fill synthetic --profile-steps 1
  NCU_COUNT=12 profiling/ncu_kernel.sh deep E6_int8_kernels - '.' \
    decode.run --mode compiled_graph --quant int8 --context 2048 --fill synthetic --profile-steps 1
  NCU_COUNT=12 profiling/ncu_kernel.sh deep E6_fp8_kernels - '.' \
    decode.run --mode compiled_graph --quant fp8 --context 2048 --fill synthetic --profile-steps 1 ;;
e7)          # vLLM: one engine per process
  $V --batches 1 8 32 --contexts 2048
  $V --batches 1 8 32 --contexts 2048 --enforce-eager
  $V --batches 1 --contexts 32768 130000 --max-model-len 131072
  $V --batches 1 --contexts 32768 130000 --max-model-len 131072 --kv-cache-dtype fp8
  $V --batches 32 --contexts 2048 --kv-cache-dtype fp8 ;;
e6-batch)    # investigate E6 at B=32 (int8 549 ms, int4 21 ms vs bf16 9.5 ms): where does it break, which kernels run?
  for b in 2 4 8 16; do $R --experiment E6-sweep --mode compiled_graph --quant int8 --batch $b --context 2048 --fill synthetic --steps 32; done
  for b in 8 16; do $R --experiment E6-sweep --mode compiled_graph --quant int4 --batch $b --context 2048 --fill synthetic --steps 32; done
  for q in int8 int4 fp8 bf16; do
    profiling/nsys_step.sh E6_${q}_B32 decode.run --mode compiled_graph --quant $q --batch 32 --context 2048 --fill synthetic --profile-steps 3
  done ;;
repeat)      # headline configs again in a later session (session-to-session variance)
  warm --mode compiled_graph --batch 1 --context 512
  for i in 1 2 3; do $R --experiment repeat --mode compiled_graph --batch 1 --context 512 --notes "S3 repeat of E3 headline"; done
  for i in 1 2; do $R --experiment repeat --mode eager --batch 1 --context 512 --notes "S3 repeat of E1 eager"; done ;;
*) echo "unknown block $1"; exit 1 ;;
esac
