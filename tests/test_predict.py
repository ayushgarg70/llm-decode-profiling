"""predict.py must reproduce the hand-derived numbers (notes/REPORT.md §2) and the known parameter counts."""

import pytest

from analysis import predict as P

# Architecture fields from the model's config.json.
LLAMA_31_8B = {
    "model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336,
    "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 8,
    "vocab_size": 128256, "tie_word_embeddings": False, "attention_bias": False,
}
QWEN25_7B = {
    "model_type": "qwen2", "hidden_size": 3584, "intermediate_size": 18944,
    "num_attention_heads": 28, "num_hidden_layers": 28, "num_key_value_heads": 4,
    "vocab_size": 152064, "tie_word_embeddings": False,
}

H100_BW = P.DATASHEET["H100-SXM"]["bandwidth_Bps"]
H100_PEAK = P.DATASHEET["H100-SXM"]["peak_flops"]


@pytest.fixture
def llama():
    return P.ModelShape.from_config(LLAMA_31_8B)


def test_llama_param_count_matches_published(llama):
    p = P.param_counts(llama)
    assert p.total == 8_030_261_248  # published Llama-3.1-8B parameter count
    assert p.per_layer == 218_103_808  # 41.9M attention + 176.2M MLP
    assert p.mlp_per_layer / p.per_layer == pytest.approx(0.808, abs=1e-3)


def test_qwen_param_count_matches_published():
    q = P.ModelShape.from_config(QWEN25_7B)
    assert q.qkv_bias
    assert P.param_counts(q).total == 7_615_616_512


def test_weight_bytes_and_floor(llama):
    assert P.weight_bytes_per_step(llama) == pytest.approx(15.0e9, rel=2e-3)
    assert P.floor_ms(llama, batch=1, context=0, bandwidth_Bps=H100_BW) == pytest.approx(4.48, abs=0.01)


def test_kv_cache(llama):
    assert P.kv_bytes_per_token(llama) == 131_072  # 128 KiB
    assert 4096 * P.kv_bytes_per_token(llama) == 512 * 1024**2
    assert 131072 * P.kv_bytes_per_token(llama) == 16 * 1024**3
    assert P.crossover_tokens(llama) == pytest.approx(114_441, rel=2e-3)


def test_flops_and_intensity(llama):
    assert P.matmul_flops_per_token(llama) == pytest.approx(15.0e9, rel=2e-3)
    assert P.attention_flops_per_token(llama, 4096) == 2_147_483_648
    assert P.matmul_intensity(batch=1) == 1
    assert P.matmul_intensity(batch=32) == 32
    assert P.attention_intensity(llama) == 4
    assert P.attention_intensity(P.ModelShape.from_config(QWEN25_7B)) == 7
    assert H100_PEAK / H100_BW == pytest.approx(295, abs=1)


def test_memory_bound_at_batch_1(llama):
    mem = P.floor_ms(llama, 1, 512, H100_BW)
    comp = P.compute_ms(llama, 1, 512, H100_PEAK)
    assert mem / comp > 250  # ~300x apart: bytes, not FLOPs, set the time


def test_tied_embeddings_are_read_in_full():
    tied = P.ModelShape.from_config({**LLAMA_31_8B, "tie_word_embeddings": True})
    p = P.param_counts(tied)
    assert p.lm_head == 0
    assert p.read_per_step == p.total  # the shared matrix is read in full as the output layer


def test_gpu_keys_and_ceilings_files():
    assert P.gpu_key("NVIDIA H100 80GB HBM3") == "H100-SXM"
    assert P.gpu_key("NVIDIA B200") == "B200"
    assert P.ceilings_path("H100-SXM") == "data/ceilings.json"  # post 1's file is untouched
    assert P.ceilings_path("B200") == "data/ceilings_B200.json"
    c, src = P.current_ceilings("NVIDIA B200")
    assert c["bandwidth_Bps"] > 5e12 and ("datasheet" in src or "measured" in src)  # measured 6.63, datasheet 7.7
