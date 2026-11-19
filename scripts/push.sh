#!/usr/bin/env bash
# Run on the Mac: copy the code to the GPU box (no venv, no git, no traces).
# Usage: scripts/push.sh <ssh-host>     e.g. scripts/push.sh h100
set -euo pipefail
HOST=${1:?usage: scripts/push.sh <ssh-host>}
cd "$(dirname "$0")/.."
rsync -az --delete \
  --exclude .venv --exclude .venv-vllm --exclude .git --exclude __pycache__ --exclude .pytest_cache \
  --exclude traces --exclude reports --exclude data --exclude logs \
  ./ "$HOST:~/decode-profiling/"
echo "pushed to $HOST:~/decode-profiling"
