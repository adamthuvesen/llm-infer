"""Token selection from next-token logits. Phase B is greedy only.

Kept trivial and separate so the decode loop reads cleanly and a temperature/top-p
sampler can slot in later (Phase E) without touching the engine.
"""

from __future__ import annotations

import torch


def greedy(logits: torch.Tensor) -> int:
    """The argmax token id from a 1-D ``(vocab_size,)`` logit row."""
    if logits.ndim != 1:
        raise ValueError(f"expected 1-D logits, got shape {tuple(logits.shape)}")
    return int(torch.argmax(logits).item())
