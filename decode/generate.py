"""Prefill + greedy decode on top of decode.model.Transformer.

This first version is plain eager PyTorch. The compiled / CUDA-graph modes and the timing hooks are
built on top of decode_one_token / decode_n_tokens, so those stay the single source of truth.
"""

from __future__ import annotations

import torch

from decode.model import Transformer


@torch.no_grad()
def prefill(model: Transformer, prompt: torch.Tensor) -> torch.Tensor:
    """Run the whole prompt [B, T] in one forward pass, filling the KV cache at positions 0..T-1.
    Returns the first generated token, [B, 1]."""
    T = prompt.shape[1]
    input_pos = torch.arange(T, device=prompt.device)
    logits = model(prompt, input_pos, last_only=True)
    return logits[:, -1].argmax(dim=-1, keepdim=True)


@torch.no_grad()
def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    """One decode step: run the model on the current token, pick the next one greedily.

    x: [B, 1] current tokens; input_pos: LongTensor [1] on the model's device. Returns [B, 1].
    Only the last position's logits are computed (lm_head on 1 token), argmax stays on the GPU,
    and nothing syncs with the CPU. This is the function wrapped in torch.compile / CUDA graphs.
    """
    logits = model(x, input_pos, last_only=True)
    return logits[:, -1].argmax(dim=-1, keepdim=True)


@torch.no_grad()
def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, n: int,
                    step_fn=None) -> list[torch.Tensor]:
    """The decode loop: n steps, each output token fed back in. Returns n tokens, each [B, 1].

    cur_token: [B, 1] from prefill, at position `input_pos` (LongTensor [1] on the model's device).
    step_fn: the step to call (default decode_one_token; later the compiled / CUDA-graph version).

    CUDA-graph rules this loop follows:
      - `input_pos` stays the same tensor and is advanced in place (a graph records addresses).
      - no CPU <-> GPU sync inside the loop.
      - a graph replay overwrites its output buffer, so each token is cloned right away; the clone
        is both what we keep and what we feed into the next step (as gpt-fast does).
    """
    step_fn = step_fn or decode_one_token
    tokens = []
    for _ in range(n):
        cur_token = step_fn(model, cur_token, input_pos).clone()
        input_pos += 1
        tokens.append(cur_token)
    return tokens


@torch.no_grad()
def generate(model: Transformer, prompt: torch.Tensor, n_new: int) -> torch.Tensor:
    """Greedy generation: prefill, then n_new - 1 decode steps. Returns [B, n_new] new tokens.
    The caller must have called model.setup_caches(batch, max_seq) with max_seq >= T + n_new."""
    first = prefill(model, prompt)
    input_pos = torch.tensor([prompt.shape[1]], device=prompt.device)
    rest = decode_n_tokens(model, first, input_pos, n_new - 1)
    return torch.cat([first, *rest], dim=1)
