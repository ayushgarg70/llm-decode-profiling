"""Load Hugging Face Llama/Qwen2 weights into decode.model.Transformer (names match, no renaming)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from decode.model import ModelArgs, Transformer

# Only what we need. The Llama repos also ship `original/consolidated.00.pth`, a second 16 GB copy
# of the weights in Meta's format; skip it.
ALLOW = ["config.json", "generation_config.json", "*.safetensors", "model.safetensors.index.json",
         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def resolve(path_or_repo: str) -> Path:
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo, allow_patterns=ALLOW))


def load_model(path_or_repo: str, device: str | torch.device = "cuda", dtype=torch.bfloat16) -> Transformer:
    """Build the model on the meta device (no memory), then load the safetensors straight onto `device`."""
    from safetensors import safe_open

    path = resolve(path_or_repo)
    args = ModelArgs.from_hf_config(json.loads((path / "config.json").read_text()))
    with torch.device("meta"):
        model = Transformer(args)
    state = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, framework="pt", device=str(device)) as sf:
            for k in sf.keys():
                state[k] = sf.get_tensor(k).to(dtype)
    if args.tie_embeddings:
        state.setdefault("lm_head.weight", state["model.embed_tokens.weight"])
    model.load_state_dict(state, strict=True, assign=True)
    if args.tie_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval().requires_grad_(False)


def from_hf_model(hf_model, device="cpu", dtype=torch.float32) -> Transformer:
    """Copy an in-memory HF model's weights into ours (used by the parity tests)."""
    args = ModelArgs.from_hf_config(hf_model.config.to_dict())
    model = Transformer(args).to(device=device, dtype=dtype)
    model.load_state_dict(hf_model.state_dict(), strict=True)
    return model.eval().requires_grad_(False)
