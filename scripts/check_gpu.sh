#!/usr/bin/env bash
# Run on the GPU box after `bootstrap_gpu.sh env`. The go / no-go check for a provider:
#   1. which GPU exactly (SXM vs PCIe), driver, CUDA, clocks, power
#   2. are nsys and ncu installed
#   3. can ncu read hardware counters (the ERR_NVGPUCTRPERM problem)
#   4. does nsys capture a trace
# Everything is also saved to data/env/check_<timestamp>.txt.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"   # uv lives here; non-interactive ssh skips .bashrc
cd "$(dirname "$0")/.."
mkdir -p data/env
OUT="data/env/check_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee "$OUT") 2>&1

section() { echo; echo "=== $* ==="; }
PROBE='import torch; x = torch.randn(1 << 26, device="cuda", dtype=torch.bfloat16); torch.cuda.synchronize(); y = x.sum(); torch.cuda.synchronize(); print("probe ok", float(y) == float(y))'

section "1. GPU, driver, clocks, power"
nvidia-smi --query-gpu=name,pci.bus_id,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem --format=csv
nvidia-smi | head -4 | tail -1   # line with driver + CUDA version
nvidia-smi -q -d CLOCK | sed -n '1,40p'
nvidia-smi -q -d POWER | grep -iE "power (limit|draw)|Default|Enforced" | head -8
uname -r; grep PRETTY_NAME /etc/os-release
uv run python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__)"
# Blackwell needs a PyTorch build that includes its architecture (sm_100) in the compiled kernels.
uv run python -c "import torch; cap = torch.cuda.get_device_capability(); arch = f'sm_{cap[0]}{cap[1]}'; archs = torch.cuda.get_arch_list(); print('compute capability', cap, '| torch built for', archs); assert arch in archs or f'compute_{cap[0]}{cap[1]}' in archs, f'torch has no kernels for {arch}'"

section "2. Nsight tools"
for t in nsys ncu; do
  P=$(command -v $t || ls -d /usr/local/cuda*/bin/$t /opt/nvidia/nsight-*/*/$t /opt/nvidia/nsight-*/$t 2>/dev/null | head -1)
  if [ -n "$P" ]; then echo "$t: $P"; "$P" --version | tail -1; else echo "$t: NOT FOUND"; fi
done
echo "RmProfilingAdminOnly (1 = only root may read counters):"
grep -i RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null || echo "  (param not readable)"

section "3. ncu hardware counters"
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
if [ -z "$NCU" ]; then
  echo "SKIP: ncu not installed"
else
  CMD=("$NCU" -k regex:reduce -c 1 --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed .venv/bin/python -c "$PROBE")
  if "${CMD[@]}" 2>&1 | tee /tmp/ncu_check.txt | grep -q "ERR_NVGPUCTRPERM"; then
    echo "!! ERR_NVGPUCTRPERM as normal user. Retrying with sudo..."
    if sudo -E "${CMD[@]}" 2>&1 | tee /tmp/ncu_check_sudo.txt | grep -q "dram__throughput"; then
      echo "RESULT: ncu works WITH sudo (fine: we run ncu via sudo -E)."
    else
      echo "RESULT: ncu FAILS even with sudo. No-go for this provider unless the driver option can be changed."
    fi
  elif grep -q "dram__throughput" /tmp/ncu_check.txt; then
    cat /tmp/ncu_check.txt | grep -E "reduce|gpu__time|dram__throughput"
    echo "RESULT: ncu works as normal user."
  else
    cat /tmp/ncu_check.txt | tail -20
    echo "RESULT: ncu gave no metrics (see output above)."
  fi
fi

section "4. nsys trace"
NSYS=$(command -v nsys || ls /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
if [ -z "$NSYS" ]; then
  echo "SKIP: nsys not installed"
else
  "$NSYS" profile -o /tmp/nsys_check --force-overwrite true --trace=cuda,nvtx .venv/bin/python -c "$PROBE" >/tmp/nsys_check.log 2>&1
  if [ -f /tmp/nsys_check.nsys-rep ]; then echo "RESULT: nsys works ($(du -h /tmp/nsys_check.nsys-rep | cut -f1) report)"; else tail -20 /tmp/nsys_check.log; echo "RESULT: nsys FAILED"; fi
fi

echo; echo "Saved to $OUT"
