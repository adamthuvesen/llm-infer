"""Single-request greedy decode: the end-to-end path the oracle validates.

No batching, no KV-cache — each step re-runs the full forward over the growing
sequence and takes the argmax. Slow, but exact and obviously correct, which is the
whole point of Phase A.
"""

from __future__ import annotations

import torch

from llm_infer.model.qwen import QwenModel


def greedy_decode(
    model: QwenModel,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> list[int]:
    """Greedily generate up to ``max_new_tokens`` continuation tokens.

    Returns only the generated ids (the prompt is not included). Stops early when an
    EOS id is produced; the EOS token itself is included in the returned list, mirror-
    ing HuggingFace ``generate`` so token-for-token comparison lines up.
    """
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1; got {max_new_tokens}")

    tokens = list(prompt_ids)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits = model.logits(tokens)
        next_id = int(torch.argmax(logits[-1]).item())
        tokens.append(next_id)
        generated.append(next_id)
        if next_id in eos_token_ids:
            break
    return generated
