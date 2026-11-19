"""Correctness on the real model (GPU): our loop vs Hugging Face, greedy, same 512-token prompt, BF16.

Reports:
  - first index where the generated tokens differ (BF16 rounding can flip a near-tie later on;
    that's expected and gets documented, not hidden)
  - teacher-forced agreement: HF's argmax at every position of (prompt + our tokens) vs our token.
    Unlike the free-running comparison, one flipped token doesn't cascade here.
  - max |logit difference| on the prompt's last position.

    python -m decode.parity_8b --new-tokens 64
"""

from __future__ import annotations

import argparse

import torch

from decode.generate import generate
from decode.hf_baseline import build_hf
from decode.load import load_model, resolve
from decode.run import DEFAULT_MODEL, make_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=64)
    args = ap.parse_args()
    dev, dt, P, N = torch.device("cuda"), torch.bfloat16, args.context, args.new_tokens

    hf, tok = build_hf(args.model, dev, dt)
    prompt = make_prompt(tok, 1, P, hf.config.vocab_size, dev)
    with torch.no_grad():
        hf_tokens = hf.generate(prompt, max_new_tokens=N, min_new_tokens=N, do_sample=False,
                                pad_token_id=tok.eos_token_id, attention_mask=torch.ones_like(prompt))[0, P:]
        hf_last = hf(prompt).logits[0, -1].float()

        ours = load_model(str(resolve(args.model)), device=dev, dtype=dt)
        ours.setup_caches(1, P + N + 1, device=dev, dtype=dt)
        our_tokens = generate(ours, prompt, N)[0]
        ours.setup_caches(1, P + 1, device=dev, dtype=dt)
        our_last = ours(prompt, torch.arange(P, device=dev), last_only=True)[0, -1].float()

        full = torch.cat([prompt[0], our_tokens[:-1]])[None]
        hf_next = hf(full).logits[0, P - 1:].argmax(-1)

    diff = (hf_tokens != our_tokens).nonzero()
    first = int(diff[0]) if len(diff) else None
    agree = (hf_next == our_tokens).float().mean().item()
    print(f"free-running: {'identical for all ' + str(N) if first is None else f'first difference at token {first}'}")
    print(f"teacher-forced agreement: {100 * agree:.1f}% of {N} positions")
    print(f"prompt last-position logits: max |diff| = {(hf_last - our_last).abs().max().item():.4f}, "
          f"same argmax: {bool(hf_last.argmax() == our_last.argmax())}")
    print("HF  :", repr(tok.decode(hf_tokens)))
    print("ours:", repr(tok.decode(our_tokens)))


if __name__ == "__main__":
    main()
