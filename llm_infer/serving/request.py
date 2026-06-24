"""A single generation request and its mutable runtime state.

Created by the caller with its prompt, stop config, and :class:`SamplingParams`; the engine
fills in the block table on admission and grows ``generated`` one token per step. Stop
semantics mirror the Phase A ``greedy_decode`` exactly (EOS token included, capped at
``max_new_tokens``) so the cached/paged path reproduces the full-recompute reference
token-for-token.

The request also owns its sampling RNG: a single :class:`torch.Generator` seeded once with
``sampling.seed`` and reused across the request's own decode steps via :meth:`generator`. A
request's draw therefore depends only on its seed and its own decode history — never on which
other requests share the batch — which is what makes a sampled request reproduce its tokens
identically run alone or batched (batched == serial under sampling).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.serving.sampler import GREEDY, SamplingParams


@dataclass
class Request:
    """One generation request, carrying its sampling params, KV block table, and output."""

    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    eos_token_ids: frozenset[int]
    prefix_group_id: str | None = None
    sampling: SamplingParams = GREEDY

    block_table: BlockTable | None = None
    prompt_cached_tokens: int = 0
    prefilled: bool = False
    finished: bool = False
    _generated_tokens: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _generated_cache: list[int] | None = field(default_factory=list, init=False, repr=False)
    _generator: torch.Generator | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1; got {self.max_new_tokens}")

    def generator(self, device: torch.device) -> torch.Generator:
        """This request's seeded RNG, created once on first sample on the logits' device.

        Seeded with ``sampling.seed`` and never re-seeded, so successive draws form one stream
        keyed only to this request's seed and its own decode steps. The device is fixed for an
        engine instance; a later device change is a real bug (CPU/CUDA mix) and raises rather
        than silently re-seeding mid-generation.
        """
        if self._generator is None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.sampling.seed)
        elif self._generator.device != torch.device(device):
            raise ValueError(
                f"request {self.request_id!r} generator is on {self._generator.device}, "
                f"got logits on {device}"
            )
        return self._generator

    @property
    def generated(self) -> list[int]:
        """Generated ids materialized as Python ints at output/test boundaries."""
        if self._generated_cache is None:
            if not self._generated_tokens:
                self._generated_cache = []
            else:
                stacked = torch.stack([token.reshape(()) for token in self._generated_tokens])
                self._generated_cache = [int(token) for token in stacked.cpu().tolist()]
        return list(self._generated_cache)

    @property
    def last_token(self) -> int:
        """The most recently produced token as a Python int, for boundary callers."""
        if not self._generated_tokens:
            raise ValueError(f"request {self.request_id!r} has produced no tokens yet")
        return self.generated[-1]

    @property
    def last_token_tensor(self) -> torch.Tensor:
        """The most recently produced token, still on its original device."""
        if not self._generated_tokens:
            raise ValueError(f"request {self.request_id!r} has produced no tokens yet")
        return self._generated_tokens[-1]

    @property
    def remaining_tokens(self) -> int:
        """How many more tokens may be emitted before the max-new-token cap."""
        return self.max_new_tokens - len(self._generated_tokens)

    @property
    def recompute_prompt_ids(self) -> list[int]:
        """The sequence a resume re-prefills: prompt plus every generated token but the last.

        At preemption the request's KV covered ``prompt + generated`` and its *next* decode was
        about to feed ``generated[-1]`` to produce the following token. Recompute must rebuild
        exactly that pre-feed state — KV for ``prompt + generated[:-1]`` — so the resuming decode
        re-feeds ``generated[-1]`` at its original position and continues identically. Writing the
        last generated token into the cache here instead would double it and shift every later
        position, breaking token-exactness.
        """
        return self.prompt_ids + self.generated[:-1]

    def reset_for_recompute(self) -> None:
        """Drop cached-KV state for preemption, keeping generated tokens for later recompute.

        Frees nothing itself (the engine frees the block table at the allocator boundary so
        the trace stays honest); it only clears the request's view of its cache so a fresh
        prefill over :attr:`recompute_prompt_ids` rebuilds it from scratch on resume.
        """
        if self.finished:
            raise ValueError(f"cannot preempt finished request {self.request_id!r}")
        self.block_table = None
        self.prompt_cached_tokens = 0
        self.prefilled = False

    def record(self, token_id: int | torch.Tensor, *, is_eos: bool | None = None) -> None:
        """Append a sampled token and apply the stop rule (EOS or length cap)."""
        if self.finished:
            raise ValueError(f"request {self.request_id!r} is already finished")
        token = torch.as_tensor(token_id, dtype=torch.long).reshape(())
        self._generated_tokens.append(token.detach())
        self._generated_cache = None

        if is_eos is None:
            is_eos = int(token.cpu().item()) in self.eos_token_ids
        if is_eos or len(self._generated_tokens) >= self.max_new_tokens:
            self.finished = True
