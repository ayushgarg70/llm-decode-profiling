"""E6 plumbing on CPU: byte accounting and INT8 weight-only on the tiny model (INT4/FP8 kernels are CUDA-only)."""

import math

import torch

from analysis.ppl import perplexity, windows_of
from analysis.predict import ModelShape, weight_bytes_per_step
from decode.quant import quantize, tensor_bytes, weight_bytes_read_per_step
from decode.run import TINY_CONFIG, build


def tiny():
    model, _ = build("tiny", torch.device("cpu"), torch.float32)
    return model


def test_bf16_bytes_match_prediction():
    model, _ = build("tiny", torch.device("cpu"), torch.bfloat16)
    assert weight_bytes_read_per_step(model) == weight_bytes_per_step(ModelShape.from_config(TINY_CONFIG))


def test_int8_weight_only_quantizes_every_linear_and_shrinks_bytes():
    model = tiny()
    before = weight_bytes_read_per_step(model)  # fp32 on CPU: 4 B/weight
    names = quantize(model, "int8")
    assert "lm_head" in names and any("mlp.down_proj" in n for n in names)
    assert "embed_tokens" not in " ".join(names)
    after = weight_bytes_read_per_step(model)
    # 1 byte per weight + one scale per output row (+ the unquantized norms)
    assert before / 4.5 < after < before / 3.5
    assert tensor_bytes(model.lm_head.weight) < model.lm_head.weight.numel() * 1.1


def test_int8_logits_close_to_float():
    ref, q = tiny(), tiny()
    quantize(q, "int8")
    x = torch.randint(0, TINY_CONFIG["vocab_size"], (1, 16))
    pos = torch.arange(16)
    for m in (ref, q):
        m.setup_caches(1, 16)
    with torch.no_grad():
        a, b = ref(x, pos), q(x, pos)
    assert torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0) > 0.999


def test_perplexity_windows_and_value():
    ids = torch.arange(100)
    w = windows_of(ids, 10, bos=1, limit=None)
    assert len(w) == 11 and all(len(x) == 10 for x in w) and w[0][0] == 1 and w[1][1] == 9
    model = tiny()
    nll, n = perplexity(model, windows_of(torch.randint(0, 256, (64,)), 16, None, None), torch.device("cpu"))
    assert n == 4 * 15 and 0 < nll < 2 * math.log(256)


def test_nvfp4_weight_only_bytes_and_logits():
    """NVFP4 = 4-bit values + one FP8 scale per 16 weights (+ a per-tensor scale): ~0.5625 B/weight."""
    ref, q = tiny(), tiny()
    quantize(q, "nvfp4w")
    w = q.lm_head.weight
    assert abs(tensor_bytes(w) / w.shape.numel() - 0.5625) < 0.01
    x = torch.randint(0, TINY_CONFIG["vocab_size"], (1, 16))
    pos = torch.arange(16)
    for m in (ref, q):
        m.setup_caches(1, 16)
    with torch.no_grad():
        a, b = ref(x, pos), q(x, pos)
    assert torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0) > 0.98


def test_nvfp4_w4a4_needs_blackwell():
    import pytest

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
        pytest.skip("on Blackwell: exercised by runs/b1.sh instead")
    with pytest.raises(Exception):
        quantize(tiny(), "nvfp4")
