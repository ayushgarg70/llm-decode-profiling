"""First-principles predictions for one decode step (notes/REPORT.md §2).

Everything is derived from a model's config.json plus the GPU's ceilings
(memory bandwidth and peak FLOP/s), so switching models or using measured
ceilings from E0 is automatic.

Usage:
    uv run python -m analysis.predict                      # Llama-3.1-8B, datasheet H100 SXM
    uv run python -m analysis.predict --config path/to/config.json --batch 32 --context 2048
    uv run python -m analysis.predict --ceilings data/ceilings.json   # measured E0 ceilings
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

KIB = 1024
GIB = 1024**3

# Datasheet ceilings (dense BF16, no sparsity). Measured E0 values replace these.
DATASHEET = {
    "H100-SXM": {"bandwidth_Bps": 3.35e12, "peak_flops": 989.4e12},
    "H100-PCIe": {"bandwidth_Bps": 2.0e12, "peak_flops": 756e12},
    # HGX B200 (180 GB HBM3e): ~7.7 TB/s, ~2.25 PFLOPS dense BF16 (post 2 spec, §13.3). Fallback only:
    # the measured B0 ceilings (data/ceilings_B200.json) replace it as soon as they exist.
    "B200": {"bandwidth_Bps": 7.7e12, "peak_flops": 2.25e15},
}


def gpu_key(device_name: str) -> str:
    """Datasheet/ceilings key for a CUDA device name, e.g. 'NVIDIA B200' -> 'B200'."""
    n = device_name.upper()
    if "B200" in n:
        return "B200"
    if "H100" in n and "PCIE" in n:
        return "H100-PCIe"
    return "H100-SXM"


def ceilings_path(key: str) -> str:
    """Measured ceilings per GPU. H100 keeps the original file name (post 1 depends on it)."""
    return "data/ceilings.json" if key.startswith("H100") else f"data/ceilings_{key}.json"


def current_ceilings(device_name: str | None = None) -> tuple[dict, str]:
    """Ceilings for the GPU we're running on (measured if available, else datasheet)."""
    if device_name is None:
        import torch

        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "H100-SXM"
    key = gpu_key(device_name)
    return load_ceilings(ceilings_path(key), key)


@dataclass(frozen=True)
class ModelShape:
    layers: int
    hidden: int
    heads: int  # query heads
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tied_embeddings: bool
    qkv_bias: bool  # Qwen2 has biases on q/k/v; Llama doesn't

    @classmethod
    def from_config(cls, cfg: dict) -> "ModelShape":
        heads = cfg["num_attention_heads"]
        return cls(
            layers=cfg["num_hidden_layers"],
            hidden=cfg["hidden_size"],
            heads=heads,
            kv_heads=cfg.get("num_key_value_heads", heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
            intermediate=cfg["intermediate_size"],
            vocab=cfg["vocab_size"],
            tied_embeddings=cfg.get("tie_word_embeddings", False),
            qkv_bias=cfg.get("attention_bias", cfg.get("model_type") == "qwen2"),
        )


@dataclass(frozen=True)
class ParamCounts:
    attn_per_layer: int  # q, k, v, o weights (+ q/k/v biases if any)
    mlp_per_layer: int  # gate, up, down
    norms: int  # RMSNorm weights: 2 per layer + final
    embedding: int
    lm_head: int  # 0 if tied (then the embedding matrix doubles as lm_head)
    layers: int

    @property
    def per_layer(self) -> int:
        return self.attn_per_layer + self.mlp_per_layer

    @property
    def total(self) -> int:
        return self.layers * self.per_layer + self.norms + self.embedding + self.lm_head

    @property
    def matmul(self) -> int:
        """Weights used in matmuls each step: all layer matrices + the output projection."""
        out_proj = self.lm_head if self.lm_head else self.embedding  # tied: embedding is the lm_head
        return self.layers * self.per_layer + out_proj

    @property
    def read_per_step(self) -> int:
        """Weights read in full each decode step. The input embedding is a lookup (one row
        per token), so it is not read in full, unless it is tied and doubles as lm_head."""
        return self.matmul + self.norms


def param_counts(m: ModelShape) -> ParamCounts:
    q = m.hidden * m.heads * m.head_dim
    kv = 2 * m.hidden * m.kv_heads * m.head_dim
    o = m.heads * m.head_dim * m.hidden
    bias = (m.heads + 2 * m.kv_heads) * m.head_dim if m.qkv_bias else 0
    mlp = 3 * m.hidden * m.intermediate
    embedding = m.vocab * m.hidden
    return ParamCounts(
        attn_per_layer=q + kv + o + bias,
        mlp_per_layer=mlp,
        norms=(2 * m.layers + 1) * m.hidden,
        embedding=embedding,
        lm_head=0 if m.tied_embeddings else embedding,
        layers=m.layers,
    )


def weight_bytes_per_step(m: ModelShape, bytes_per_weight: float = 2.0) -> float:
    return param_counts(m).read_per_step * bytes_per_weight


def kv_bytes_per_token(m: ModelShape, kv_bytes: float = 2.0) -> float:
    """K and V, every layer, every KV head, head_dim numbers each."""
    return 2 * m.layers * m.kv_heads * m.head_dim * kv_bytes


def crossover_tokens(m: ModelShape, bytes_per_weight: float = 2.0, kv_bytes: float = 2.0) -> float:
    """B × s at which KV-cache reads equal weight reads."""
    return weight_bytes_per_step(m, bytes_per_weight) / kv_bytes_per_token(m, kv_bytes)


def matmul_flops_per_token(m: ModelShape) -> float:
    return 2 * param_counts(m).matmul


def attention_flops_per_token(m: ModelShape, context: int) -> float:
    """Scores (2·d·s) + weighted sum of V (2·d·s), per query head per layer."""
    return 4 * m.layers * m.heads * m.head_dim * context


def matmul_intensity(batch: int, bytes_per_weight: float = 2.0) -> float:
    """FLOPs per byte: each weight does 2 FLOPs per request and is read once per step."""
    return 2 * batch / bytes_per_weight


def attention_intensity(m: ModelShape, kv_bytes: float = 2.0) -> float:
    """4·H·d·s FLOPs over 2·H_kv·d·s·kv_bytes bytes; independent of batch and context."""
    return (4 * m.heads) / (2 * m.kv_heads * kv_bytes)


def step_bytes(m: ModelShape, batch: int, context: int, bytes_per_weight: float = 2.0, kv_bytes: float = 2.0) -> float:
    return weight_bytes_per_step(m, bytes_per_weight) + batch * context * kv_bytes_per_token(m, kv_bytes)


def step_flops(m: ModelShape, batch: int, context: int) -> float:
    return batch * (matmul_flops_per_token(m) + attention_flops_per_token(m, context))


def floor_ms(m: ModelShape, batch: int, context: int, bandwidth_Bps: float,
             bytes_per_weight: float = 2.0, kv_bytes: float = 2.0) -> float:
    """Memory floor for one decode step: bytes read ÷ bandwidth."""
    return 1e3 * step_bytes(m, batch, context, bytes_per_weight, kv_bytes) / bandwidth_Bps


def compute_ms(m: ModelShape, batch: int, context: int, peak_flops: float) -> float:
    return 1e3 * step_flops(m, batch, context) / peak_flops


def load_config(source: str) -> dict:
    """A local config.json path, or a Hugging Face repo id (needs `hf auth login` for gated repos)."""
    path = Path(source)
    if path.exists():
        return json.loads(path.read_text())
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(source, "config.json")).read_text())


def load_ceilings(path: str | None, gpu: str) -> tuple[dict, str]:
    if path and Path(path).exists():
        c = json.loads(Path(path).read_text())
        return {"bandwidth_Bps": c["bandwidth_Bps"], "peak_flops": c["peak_flops"]}, f"measured ({path})"
    return DATASHEET[gpu], f"datasheet {gpu}"


def report(m: ModelShape, ceilings: dict, source: str, batch: int, context: int) -> str:
    p = param_counts(m)
    bw, peak = ceilings["bandwidth_Bps"], ceilings["peak_flops"]
    kv_tok = kv_bytes_per_token(m)
    lines = [
        f"Ceilings: {bw / 1e12:.3f} TB/s, {peak / 1e12:.1f} TFLOPS  [{source}]",
        f"Params total {p.total / 1e9:.3f}B | per layer {p.per_layer / 1e6:.1f}M "
        f"(attn {p.attn_per_layer / 1e6:.1f}M, MLP {p.mlp_per_layer / 1e6:.1f}M) | "
        f"embedding {p.embedding / 1e9:.3f}B | lm_head {p.lm_head / 1e9:.3f}B",
        f"Weights read per step: {p.read_per_step / 1e9:.3f}B params = {weight_bytes_per_step(m) / 1e9:.2f} GB (BF16)",
        f"Batch-1 floor (s=0): {floor_ms(m, 1, 0, bw):.3f} ms/token = {1e3 / floor_ms(m, 1, 0, bw):.1f} tok/s",
        f"KV cache per token: {kv_tok:,.0f} B = {kv_tok / KIB:.0f} KiB",
        "KV per sequence: " + ", ".join(f"{s // 1024}k → {s * kv_tok / GIB:g} GiB" for s in (4096, 32768, 131072)),
        f"Crossover B×s (KV = weights): {crossover_tokens(m):,.0f} tokens",
        f"Matmul FLOPs/token: {matmul_flops_per_token(m) / 1e9:.2f} G | "
        f"attention FLOPs/token: {attention_flops_per_token(m, 1) / 1e6:.3f} M × s",
        f"Intensity: weight matmuls = B FLOPs/byte; decode attention = {attention_intensity(m):g} FLOPs/byte",
        f"Ridge point: {peak / bw:.0f} FLOPs/byte",
        "",
        f"At B={batch}, s={context}: bytes {step_bytes(m, batch, context) / 1e9:.2f} GB, "
        f"floor {floor_ms(m, batch, context, bw):.3f} ms/step, compute {compute_ms(m, batch, context, peak):.3f} ms/step, "
        f"{batch / floor_ms(m, batch, context, bw) * 1e3:,.0f} tok/s at the floor",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="meta-llama/Llama-3.1-8B", help="config.json path or HF repo id")
    ap.add_argument("--ceilings", default="data/ceilings.json", help="measured E0 ceilings (JSON), if present")
    ap.add_argument("--gpu", default="H100-SXM", choices=sorted(DATASHEET), help="datasheet fallback")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--context", type=int, default=512)
    args = ap.parse_args()
    m = ModelShape.from_config(load_config(args.config))
    ceilings, source = load_ceilings(args.ceilings, args.gpu)
    print(report(m, ceilings, source, args.batch, args.context))


if __name__ == "__main__":
    main()
