"""Record the software and hardware a run used (versions, GPU variant, driver, clocks, power)."""

from __future__ import annotations

import importlib.metadata as md
import shutil
import subprocess

import torch


def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return ""


def smi_query(fields: str) -> list[str]:
    """One row of `nvidia-smi --query-gpu` for GPU 0, or [] if there's no nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []
    out = subprocess.run(
        ["nvidia-smi", "-i", "0", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    return [x.strip() for x in out.split(",")] if out else []


def form_factor(name: str) -> str:
    """H100 SXM reports as e.g. 'NVIDIA H100 80GB HBM3'; PCIe and NVL say so in the name."""
    n = name.upper()
    if "PCIE" in n:
        return "PCIe"
    if "NVL" in n:
        return "NVL"
    if "SXM" in n or "HBM3" in n or "B200" in n or "B300" in n:  # B200/B300 only ship as SXM modules
        return "SXM"
    return "unknown"


def environment() -> dict:
    env = {
        "torch": torch.__version__,
        "transformers": _version("transformers"),
        "torchao": _version("torchao"),
        "cuda": torch.version.cuda or "",
    }
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        smi = smi_query("driver_version")
        env.update(gpu=name, gpu_form_factor=form_factor(name), driver=smi[0] if smi else "")
    else:
        env.update(gpu="cpu", gpu_form_factor="", driver="")
    return env


if __name__ == "__main__":
    for k, v in environment().items():
        print(f"{k}: {v}")
