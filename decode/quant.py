"""Weight quantization for E6, via torchao (written against torchao 0.18; its API names change between
versions, so this file is the single place that knows them).

Schemes (all applied to every nn.Linear, including lm_head; the embedding stays BF16 because a decode
step reads only one row of it):
    int8   weight-only, symmetric, one scale per output row.  Linear = mm(x, w_int8.to(bf16)) * scale.
           In eager that materialises a BF16 copy of the weight; under torch.compile Inductor fuses the
           cast into the matmul, so only the int8 bytes are read.  -> measure compiled modes.
    int4   weight-only, groups of 128 along K with a BF16 scale + zero point per group, "tile_packed_to_4d"
           layout -> PyTorch's built-in tinygemm kernel (aten._weight_int4pack_mm; the gpt-fast kernel).
    fp8    FP8 e4m3 weights AND activations (activations quantized on the fly), one scale per row
           -> torch._scaled_mm, i.e. cuBLAS FP8 GEMMs on Hopper's FP8 tensor cores (CUDA, sm89+ only).
"""

from __future__ import annotations

import torch
from torch import nn

SCHEMES = ("bf16", "int8", "int4", "fp8")


def config_for(scheme: str):
    import torchao.quantization as q

    if scheme == "int8":
        return q.Int8WeightOnlyConfig()
    if scheme == "int4":
        return q.Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d")
    if scheme == "fp8":
        from torchao.quantization.granularity import PerRow
        from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

        return q.Float8DynamicActivationFloat8WeightConfig(granularity=PerRow(), kernel_preference=KernelPreference.TORCH)
    raise ValueError(scheme)


def quantize(model: nn.Module, scheme: str) -> list[str]:
    """Quantize every nn.Linear in place. Returns the names of the modules that were quantized."""
    if scheme == "bf16":
        return []
    from torchao.quantization import quantize_

    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    quantize_(model, config_for(scheme))
    return names


def tensor_bytes(t: torch.Tensor) -> int:
    """Bytes actually stored for a tensor; for torchao tensor subclasses, the sum of their inner tensors
    (packed values + scales + zero points)."""
    if hasattr(t, "__tensor_flatten__") and type(t) not in (torch.Tensor, nn.Parameter):
        inner, _ = t.__tensor_flatten__()
        return sum(tensor_bytes(getattr(t, name)) for name in inner)
    return t.numel() * t.element_size()


def weight_bytes_read_per_step(model: nn.Module) -> int:
    """Bytes of parameters one decode step reads: everything except the input embedding (only one row
    per token is gathered), unless it is tied to lm_head."""
    tied = getattr(getattr(model, "args", None), "tie_embeddings", False)
    total = 0
    for name, p in model.named_parameters():
        if "embed_tokens" in name and not tied:
            continue
        total += tensor_bytes(p)
    return total
