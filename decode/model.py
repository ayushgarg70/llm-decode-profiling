"""A minimal Llama decoder in plain PyTorch, built for profiling decode.

Written in the style of PyTorch's gpt-fast (https://github.com/pytorch-labs/gpt-fast), with three
deliberate differences that matter for the measurements:

1. Parameter names match Hugging Face's LlamaForCausalLM, so HF safetensors load with no renaming.
2. GQA uses SDPA's `enable_gqa=True` instead of `repeat_interleave`, so K/V are not copied up to all
   query heads every step (a copy would read and write ~4x more KV bytes).
3. The causal mask is computed from positions on the fly, not stored as a [max_seq, max_seq] table
   (which would be 16 GiB of booleans at 128k context).

The KV cache is static: allocated once by `setup_caches(batch, max_seq)` and written in place, so a
decode step never changes tensor shapes or addresses. That is what makes CUDA graphs possible later.

Note: masked attention over a static cache reads all `max_seq` positions every step, not just the
filled ones. Size `max_seq` to the context you actually measure (a masked attention reads the whole cache).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    norm_eps: float
    max_position: int
    qkv_bias: bool = False
    tie_embeddings: bool = False
    hf_config: dict = field(default_factory=dict, repr=False)  # raw config.json, used for RoPE

    @classmethod
    def from_hf_config(cls, cfg: dict) -> "ModelArgs":
        heads = cfg["num_attention_heads"]
        return cls(
            dim=cfg["hidden_size"],
            n_layers=cfg["num_hidden_layers"],
            n_heads=heads,
            n_kv_heads=cfg.get("num_key_value_heads", heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
            intermediate=cfg["intermediate_size"],
            vocab=cfg["vocab_size"],
            norm_eps=cfg.get("rms_norm_eps", 1e-6),
            max_position=cfg.get("max_position_embeddings", 4096),
            qkv_bias=cfg.get("attention_bias", cfg.get("model_type") == "qwen2"),
            tie_embeddings=cfg.get("tie_word_embeddings", False),
            hf_config=dict(cfg),
        )


def rope_inv_freq(args: ModelArgs) -> tuple[torch.Tensor, float]:
    """Inverse frequencies for RoPE, computed by Hugging Face's own code.

    Llama-3.1 uses `rope_type: "llama3"`, which rescales the low frequencies. It changes the rotation
    at every position, so getting it wrong breaks parity with HF even on short prompts. Reusing HF's
    implementation avoids re-deriving it; tests/test_parity.py checks the result.
    """
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    keys = ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
            "max_position_embeddings", "rope_theta", "rope_scaling", "rope_parameters")
    cfg = LlamaConfig(**{k: v for k, v in args.hf_config.items() if k in keys})
    rot = LlamaRotaryEmbedding(cfg)
    return rot.inv_freq.float(), float(rot.attention_scaling)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Same convention and precision as HF (rotate-half; cos/sin already in the model dtype)."""
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class KVCache(nn.Module):
    """Static, preallocated KV cache for one layer: shape [batch, n_kv_heads, max_seq, head_dim]."""

    def __init__(self, batch: int, n_kv_heads: int, max_seq: int, head_dim: int, dtype, device=None):
        super().__init__()
        shape = (batch, n_kv_heads, max_seq, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device), persistent=False)

    def update(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Write the new keys/values at positions `input_pos` and return the full caches.

        input_pos: LongTensor [T] (T = prompt length in prefill, 1 in decode), on the cache's device.
        k, v: [batch, n_kv_heads, T, head_dim]. Returns the whole [batch, n_kv_heads, max_seq, head_dim]
        buffers; attention reads all of them and the mask hides positions not written yet.

        Written in place (`index_copy_` into the existing buffers) and indexed with the tensor itself,
        so shapes and addresses never change and nothing syncs with the CPU: both needed for CUDA graphs.
        """
        self.k_cache.index_copy_(2, input_pos, k)
        self.v_cache.index_copy_(2, input_pos, v)
        return self.k_cache, self.v_cache


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = args.n_heads, args.n_kv_heads, args.head_dim
        self.q_proj = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=args.qkv_bias)
        self.k_proj = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.qkv_bias)
        self.v_proj = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.qkv_bias)
        self.o_proj = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.kv_cache: KVCache | None = None

    def forward(self, x, cos, sin, mask, input_pos):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k, v = self.kv_cache.update(input_pos, k, v)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, enable_gqa=self.n_heads != self.n_kv_heads
        )
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.dim, args.intermediate, bias=False)
        self.up_proj = nn.Linear(args.dim, args.intermediate, bias=False)
        self.down_proj = nn.Linear(args.intermediate, args.dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = RMSNorm(args.dim, args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x, cos, sin, mask, input_pos):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, input_pos)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaBody(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab, args.dim)
        self.layers = nn.ModuleList(DecoderLayer(args) for _ in range(args.n_layers))
        self.norm = RMSNorm(args.dim, args.norm_eps)


class Transformer(nn.Module):
    """Same module tree as HF's LlamaForCausalLM: `model.embed_tokens`, `model.layers.{i}...`, `lm_head`."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = LlamaBody(args)
        self.lm_head = nn.Linear(args.dim, args.vocab, bias=False)
        if args.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.nvtx_layers = False  # per-layer NVTX ranges; eager mode only, never during timing runs
        self.max_batch = self.max_seq = 0

    def setup_caches(self, batch: int, max_seq: int, device=None, dtype=None) -> None:
        """Allocate the static KV caches and the RoPE tables. Call once, before prefill."""
        device = device or self.lm_head.weight.device
        dtype = dtype or self.lm_head.weight.dtype
        a = self.args
        for layer in self.model.layers:
            layer.self_attn.kv_cache = KVCache(batch, a.n_kv_heads, max_seq, a.head_dim, dtype, device)
        inv_freq, scale = rope_inv_freq(a)
        positions = torch.arange(max_seq, device=device)
        freqs = positions[:, None].float() * inv_freq.to(device)[None, :]  # fp32, like HF
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", (emb.cos() * scale).to(dtype), persistent=False)
        self.register_buffer("rope_sin", (emb.sin() * scale).to(dtype), persistent=False)
        self.register_buffer("positions", positions, persistent=False)
        self.max_batch, self.max_seq = batch, max_seq

    def forward(self, idx: torch.Tensor, input_pos: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        """idx: [B, T] token ids; input_pos: [T] positions. Returns logits [B, T, vocab] (or [B, 1, vocab])."""
        x = self.model.embed_tokens(idx)
        cos = self.rope_cos[input_pos][None, None]  # [1, 1, T, head_dim]
        sin = self.rope_sin[input_pos][None, None]
        mask = (self.positions[None, :] <= input_pos[:, None])[None, None]  # [1, 1, T, max_seq]
        for i, layer in enumerate(self.model.layers):
            if self.nvtx_layers:
                torch.cuda.nvtx.range_push(f"layer {i}")
            x = layer(x, cos, sin, mask, input_pos)
            if self.nvtx_layers:
                torch.cuda.nvtx.range_pop()
        if self.nvtx_layers:
            torch.cuda.nvtx.range_push("lm_head")
        x = self.model.norm(x)
        if last_only:
            x = x[:, -1:]
        logits = self.lm_head(x)
        if self.nvtx_layers:
            torch.cuda.nvtx.range_pop()
        return logits
