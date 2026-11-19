#!/usr/bin/env bash
# Run on the GPU box, from ~/decode-profiling.
#   scripts/bootstrap_gpu.sh env     # uv + Python env with CUDA PyTorch (~3-5 min)
#   scripts/bootstrap_gpu.sh vllm    # separate venv .venv-vllm for E7 (vLLM pins its own torch)
#   scripts/bootstrap_gpu.sh model   # download Llama-3.1-8B weights (~16 GB); run `hf auth login` first
# Set HF_HOME to a persistent volume (if the VM has one) so weights survive between sessions.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"   # uv installs here; non-interactive ssh shells skip .bashrc
STEP=${1:?usage: scripts/bootstrap_gpu.sh env|vllm|model}

case "$STEP" in
env)
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv sync --python 3.11
  uv run python - <<'EOF'
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda,
      "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "PyTorch can't see the GPU: check driver vs torch CUDA version"
print("GPU:", torch.cuda.get_device_name(0))
EOF
  ;;
vllm)
  # Triton (used by vLLM) compiles a C helper at start-up and needs the Python headers.
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.12-dev
  [ -d .venv-vllm ] || uv venv .venv-vllm --python 3.12
  uv pip install --python .venv-vllm/bin/python vllm pandas
  .venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
  ;;
model)
  echo "HF_HOME=${HF_HOME:-~/.cache/huggingface (default; not persistent across instances)}"
  df -h "${HF_HOME:-$HOME}" | tail -1
  uv run python -c "from decode.load import resolve; print('weights at', resolve('meta-llama/Llama-3.1-8B'))"
  ;;
*) echo "unknown step: $STEP"; exit 1 ;;
esac
