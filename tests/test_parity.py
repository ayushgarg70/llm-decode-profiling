"""Our model must match Hugging Face's, on tiny random models, in fp32 on the CPU.

The KV-cache and decode-loop tests also check the CUDA-graph rules (in-place cache writes, in-place
input_pos, cloned outputs) using a step function that mimics a graph replay.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from decode import generate as G
from decode.load import from_hf_model
from decode.model import KVCache

P, N = 12, 20  # prompt length, new tokens

LLAMA_TINY = dict(
    hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
    num_hidden_layers=2, vocab_size=256, max_position_embeddings=256, rms_norm_eps=1e-5,
    rope_theta=500000.0, tie_word_embeddings=False, attention_bias=False, initializer_range=0.1,
    # Same RoPE scaling *type* as Llama-3.1, with a small original context so all three
    # frequency bands (kept / smoothed / scaled) occur at head_dim 16.
    rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                  "high_freq_factor": 4.0, "original_max_position_embeddings": 32},
)
QWEN_TINY = dict(
    hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
    num_hidden_layers=2, vocab_size=256, max_position_embeddings=256, rope_theta=1e6,
    tie_word_embeddings=False, use_sliding_window=False, initializer_range=0.1,
)


def skip_if_unimplemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")


def build(kind: str, batch: int = 1):
    torch.manual_seed(0)
    if kind == "llama":
        hf = LlamaForCausalLM(LlamaConfig(**LLAMA_TINY))
    else:
        hf = Qwen2ForCausalLM(Qwen2Config(**QWEN_TINY))
    hf = hf.float().eval()
    ours = from_hf_model(hf)
    ours.setup_caches(batch=batch, max_seq=P + N, device="cpu", dtype=torch.float32)
    prompt = torch.randint(0, hf.config.vocab_size, (batch, P))
    return hf, ours, prompt


def test_rope_tables_match_hf():
    hf, ours, _ = build("llama")
    rot = LlamaRotaryEmbedding(hf.config)
    pos = torch.arange(P + N)[None]
    cos, sin = rot(torch.zeros(1, dtype=torch.float32), pos)
    torch.testing.assert_close(ours.rope_cos, cos[0])
    torch.testing.assert_close(ours.rope_sin, sin[0])
    # llama3 scaling really changed something vs plain RoPE (otherwise this test proves little)
    plain = 1.0 / (500000.0 ** (torch.arange(0, 16, 2).float() / 16))
    assert not torch.allclose(rot.inv_freq, plain)


def test_kv_cache_update():
    B, H, S, D = 2, 2, 8, 4
    cache = KVCache(B, H, S, D, dtype=torch.float32)
    k_buf, v_buf = cache.k_cache, cache.v_cache  # holding references, so identity checks are exact
    k0, v0 = torch.randn(B, H, 3, D), torch.randn(B, H, 3, D)
    k, v = skip_if_unimplemented(cache.update, torch.arange(3), k0, v0)  # prefill: positions 0..2
    assert cache.k_cache is k_buf and cache.v_cache is v_buf, "write in place into the existing buffers"
    k1, v1 = torch.randn(B, H, 1, D), torch.randn(B, H, 1, D)
    k, v = cache.update(torch.tensor([3]), k1, v1)  # one decode step: position 3
    assert cache.k_cache is k_buf and cache.v_cache is v_buf, "write in place into the existing buffers"
    assert k is k_buf and v is v_buf, "return the cache buffers themselves"
    assert k.shape == v.shape == (B, H, S, D)
    torch.testing.assert_close(k[:, :, :3], k0)
    torch.testing.assert_close(v[:, :, 3:4], v1)
    assert k[:, :, 4:].abs().sum() == 0, "positions not written yet stay untouched"


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
def test_prefill_logits_match_hf(kind):
    hf, ours, prompt = build(kind)
    logits = skip_if_unimplemented(ours, prompt, torch.arange(P))
    torch.testing.assert_close(logits, hf(prompt).logits, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
@pytest.mark.parametrize("batch", [1, 3])
def test_greedy_tokens_match_hf(kind, batch):
    hf, ours, prompt = build(kind, batch)
    tokens = skip_if_unimplemented(G.generate, ours, prompt, N)
    assert tokens.shape == (batch, N)
    # Teacher forcing: run HF once on prompt + our tokens; its argmax at every position must be our next token.
    full = torch.cat([prompt, tokens[:, :-1]], dim=1)
    hf_next = hf(full).logits[:, P - 1:].argmax(-1)
    assert torch.equal(hf_next, tokens)


def test_decode_loop_is_graph_friendly():
    _, ours, prompt = build("llama")
    first = skip_if_unimplemented(G.prefill, ours, prompt)

    # Reference run with the plain step function.
    expected = skip_if_unimplemented(G.decode_n_tokens, ours, first, torch.tensor([P]), 5)

    # Now a step function that behaves like a CUDA-graph replay: it always returns the SAME output
    # buffer, overwritten on every call.
    static_out = torch.zeros_like(first)

    def graph_like_step(model, x, pos):
        static_out.copy_(G.decode_one_token(model, x, pos))
        return static_out

    ours.setup_caches(batch=1, max_seq=P + N, device="cpu", dtype=torch.float32)
    first = G.prefill(ours, prompt)
    input_pos = torch.tensor([P])
    ptr = input_pos.data_ptr()
    out = G.decode_n_tokens(ours, first, input_pos, 5, step_fn=graph_like_step)
    assert input_pos.data_ptr() == ptr, "advance input_pos in place, don't replace it"
    assert input_pos.item() == P + 5
    assert all(torch.equal(a, b) for a, b in zip(out, expected)), "keep your own copy of each token"


def test_compiled_step_matches_eager():
    """torch.compile(fullgraph=True) must trace the step without graph breaks and give the same tokens."""
    _, ours, prompt = build("llama")
    first = G.prefill(ours, prompt)
    expected = G.decode_n_tokens(ours, first, torch.tensor([P]), 5)
    ours.setup_caches(batch=1, max_seq=P + N, device="cpu", dtype=torch.float32)
    first = G.prefill(ours, prompt)
    compiled = torch.compile(G.decode_one_token, fullgraph=True)
    got = G.decode_n_tokens(ours, first, torch.tensor([P]), 5, step_fn=compiled)
    assert all(torch.equal(a, b) for a, b in zip(got, expected))
