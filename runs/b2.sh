#!/usr/bin/env bash
# Second Blackwell session (post 2), same 1x B200 VM. Fills the gaps the B1 session left:
#   runs/b2.sh fixed | b4-more | repeat | ncu-free | ncu-fp8 | vllm
#   fixed     fixed cost per kernel, measured directly (bench/fixed_cost.py)
#   b4-more   context sweep: 512, 8k, 16k, 64k (B1 session had 2k, 32k, ~128k only), 3 runs each
#   repeat    a third run of every B2/B3/B4 config that has two
#   ncu-free  Nsight Compute with the GPU's clocks left alone (B1's ncu ran at base clock: DRAM % unusable),
#             on the kernels the post leans on, plus the lm_head recapture
#   vllm      vLLM reference (needs `scripts/bootstrap_gpu.sh vllm` first; VLLM_VERSION=0.34.0 as on the H100)
# Unattended:  SESSION=b2 runs/chain.sh fixed b4-more repeat ncu-free vllm
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R="$PY -m decode.run"
V=".venv-vllm/bin/python -m decode.vllm_bench"
warm() { $R "$@" --no-log >/dev/null 2>&1 || true; }   # identical args: compiles the same shape (steady state)

case "${1:?block}" in
fixed)
  $PY -m bench.fixed_cost ;;
b4-more)
  for s in 512 8192 16384 65536; do
    warm --mode compiled_graph --batch 1 --context $s --fill synthetic
    for i in 1 2 3; do $R --experiment B4 --mode compiled_graph --batch 1 --context $s --fill synthetic; done
  done ;;
repeat)
  for b in 1 8 32 128; do
    warm --mode compiled_graph --batch $b --context 2048 --fill synthetic
    $R --experiment B2 --mode compiled_graph --batch $b --context 2048 --fill synthetic
  done
  for s in 2048 32768 130794; do
    warm --mode compiled_graph --batch 1 --context $s --fill synthetic
    $R --experiment B4 --mode compiled_graph --batch 1 --context $s --fill synthetic
  done
  for q in bf16 fp8 nvfp4 nvfp4w; do
    warm --mode compiled_graph --quant $q --batch 1 --context 2048 --fill synthetic
    $R --experiment B3 --mode compiled_graph --quant $q --batch 1 --context 2048 --fill synthetic || echo "FAILED B3 $q B1"
  done
  for q in fp8 nvfp4w; do   # bf16 B=32 already has 6 runs
    warm --mode compiled_graph --quant $q --batch 32 --context 2048 --fill synthetic
    $R --experiment B3 --mode compiled_graph --quant $q --batch 32 --context 2048 --fill synthetic || echo "FAILED B3 $q B32"
  done
  # NVFP4 at B=32 only compiles with this Inductor fusion off (B1 session, b3-retry2)
  ( export TORCHINDUCTOR_NESTED_REDUCTION=0
    warm --mode compiled_graph --quant nvfp4 --batch 32 --context 2048 --fill synthetic --notes "TORCHINDUCTOR_NESTED_REDUCTION=0"
    $R --experiment B3 --mode compiled_graph --quant nvfp4 --batch 32 --context 2048 --fill synthetic --notes "TORCHINDUCTOR_NESTED_REDUCTION=0" || echo "FAILED B3 nvfp4 B32" )
  ;;
ncu-free)
  export NCU_CLOCK=none
  # the 7 BF16 matmuls of layer 0 and lm_head (225th matmul), exactly as the graph step runs them
  NCU_COUNT=7 profiling/ncu_kernel.sh deep B_bf16_matmuls_free - 'nvjet' decode.run --mode compiled_graph --context 512 --profile-steps 1
  NCU_SKIP=224 NCU_COUNT=1 profiling/ncu_kernel.sh deep B_lm_head_free - 'nvjet' decode.run --mode compiled_graph --context 512 --profile-steps 1
  NCU_COUNT=7 profiling/ncu_kernel.sh deep B_nvfp4_matmuls_free - 'nvjet' decode.run --mode compiled_graph --quant nvfp4 --context 2048 --fill synthetic --profile-steps 1 || echo "FAILED ncu nvfp4"
  NCU_COUNT=1 profiling/ncu_kernel.sh deep B_attn_64k_free "layer 5" 'sdpa|flash|fmha|attn|cudnn_generated' decode.run --mode eager --context 65536 --fill synthetic --profile-steps 1 --nvtx-layers
  ;;
ncu-fp8)     # ncu matches the short kernel name; for the CUTLASS FP8 matmuls that is "device_kernel" (not "cutlass")
  export NCU_CLOCK=none
  NCU_COUNT=7 profiling/ncu_kernel.sh deep B_fp8_matmuls_free - 'device_kernel' decode.run --mode compiled_graph --quant fp8 --context 2048 --fill synthetic --profile-steps 1
  ;;
vllm)        # one engine per process; same configs as the B200 batch and context sweeps
  # vLLM JIT-compiles some attention kernels on Blackwell: needs ninja (in the venv) and nvcc on PATH
  export PATH="$PWD/.venv-vllm/bin:/usr/local/cuda/bin:$PATH"
  $V --experiment B5 --batches 1 8 32 128 --contexts 2048
  $V --experiment B5 --batches 1 --contexts 2048 --enforce-eager
  $V --experiment B5 --batches 1 --contexts 32768 130000 --max-model-len 131072
  ;;
*) echo "unknown block $1"; exit 1 ;;
esac
