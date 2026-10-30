"""A single generation request and its mutable runtime state.

Created by the caller with its prompt, stop config, and :class:`SamplingParams`; the engine
fills in the block table on admission and grows ``generated`` one token per step. Stop
semantics mirror the reference check ``greedy_decode`` exactly (EOS token included, capped at
``max_new_tokens``) so the cached/paged path reproduces the full-recompute reference
token-for-token.

The request also owns its sampling random-number-generator (RNG): a single
:class:`torch.Generator` seeded once with ``sampling.seed`` and reused across its decode steps via
:meth:`generator`. Batchmates cannot consume this request's draws. Exact sampled tokens may still
change across batch shapes when bf16 model kernels produce slightly different logits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.serving.sampler import GREEDY, SamplingParams

PrefillMode = Literal["initial", "recompute"]


@dataclass(frozen=True)
class Pending:
    mode: PrefillMode = "initial"


@dataclass(frozen=True)
class Prefilling:
    table: BlockTable
    cached_tokens: int
    mode: PrefillMode


@dataclass(frozen=True)
class DecodeReady:
    table: BlockTable


@dataclass(frozen=True)
class Terminal:
    reason: Literal["eos", "length", "cancelled"]
    table: BlockTable | None = None


RequestState = Pending | Prefilling | DecodeReady | Terminal


@dataclass(eq=False)
class Request:
    """One generation request, carrying its sampling params, KV block table, and output."""

    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    eos_token_ids: frozenset[int]
    prefix_group_id: str | None = None
    sampling: SamplingParams = GREEDY

    _state: RequestState = field(default_factory=Pending, init=False, repr=False)
    _generated_tokens: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _generated_cache: list[int] | None = field(default_factory=list, init=False, repr=False)
    _generator: torch.Generator | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1; got {self.max_new_tokens}")

    @property
    def state(self) -> RequestState:
        return self._state

    @property
    def block_table(self) -> BlockTable | None:
        state = self._state
        return None if isinstance(state, Pending) else state.table

    @property
    def prompt_cached_tokens(self) -> int:
        state = self._state
        if isinstance(state, Prefilling):
            return state.cached_tokens
        table = self.block_table
        return table.length if table is not None else 0

    @property
    def prefilled(self) -> bool:
        return isinstance(self._state, DecodeReady)

    @property
    def finished(self) -> bool:
        return isinstance(self._state, Terminal)

    @property
    def generated_count(self) -> int:
        return len(self._generated_tokens)

    def begin_prefill(self, table: BlockTable, *, cached_tokens: int = 0) -> Prefilling:
        state = self._state
        if not isinstance(state, Pending):
            raise ValueError(f"request {self.request_id!r} is not pending")
        total = self._prefill_length(state.mode)
        if not 0 <= cached_tokens <= total or table.length != cached_tokens:
            raise ValueError("prefill progress must match the block table and sequence length")
        self._state = Prefilling(table, cached_tokens, state.mode)
        return self._state

    def advance_prefill(self, cached_tokens: int) -> None:
        state = self._state
        if not isinstance(state, Prefilling):
            raise ValueError(f"request {self.request_id!r} is not prefilling")
        if not state.cached_tokens < cached_tokens <= self._prefill_length(state.mode):
            raise ValueError("prefill progress must advance within the sequence")
        if state.table.length != cached_tokens:
            raise ValueError("prefill progress must match the block table length")
        self._state = Prefilling(state.table, cached_tokens, state.mode)

    def finish_recompute(self) -> None:
        state = self._state
        if not isinstance(state, Prefilling) or state.mode != "recompute":
            raise ValueError(f"request {self.request_id!r} is not recomputing")
        if state.cached_tokens != self._prefill_length(state.mode):
            raise ValueError("cannot finish an incomplete recompute")
        self._state = DecodeReady(state.table)

    def release_finished_table(self) -> None:
        """Remove the terminal table after the engine returns its blocks."""
        state = self._state
        if not isinstance(state, Terminal):
            raise ValueError(f"request {self.request_id!r} is not finished")
        self._state = Terminal(state.reason)

    def cancel(self) -> None:
        if self.finished:
            raise ValueError(f"request {self.request_id!r} is already finished")
        self._state = Terminal("cancelled", self.block_table)

    def _prefill_length(self, mode: PrefillMode) -> int:
        generated_prefix = self.generated_count - 1 if mode == "recompute" else 0
        return len(self.prompt_ids) + generated_prefix

    def generator(self, device: torch.device) -> torch.Generator:
        """This request's seeded RNG, created once on first sample on the logits' device.

        Seeded with ``sampling.seed`` and never re-seeded, so successive draws form one stream
        keyed to this request's seed and draw count. The device is fixed for an
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
        the trace stays clear); it only clears the request's view of its cache so a fresh
        prefill over :attr:`recompute_prompt_ids` rebuilds it from scratch on resume.
        """
        state = self._state
        if isinstance(state, Terminal):
            raise ValueError(f"cannot preempt finished request {self.request_id!r}")
        mode = "recompute" if isinstance(state, DecodeReady) else state.mode
        self._state = Pending(mode)

    def record(self, token_id: int | torch.Tensor, *, is_eos: bool | None = None) -> None:
        """Append a sampled token and apply the stop rule (EOS or length cap)."""
        state = self._state
        if isinstance(state, Terminal):
            raise ValueError(f"request {self.request_id!r} is already finished")
        if isinstance(state, (Pending, Prefilling)) and state.mode == "recompute":
            raise ValueError("cannot record a token before recompute")
        if isinstance(state, Prefilling) and state.cached_tokens != len(self.prompt_ids):
            raise ValueError("cannot record a token before initial prefill completes")
        token = torch.as_tensor(token_id, dtype=torch.long).reshape(())
        self._generated_tokens.append(token.detach())
        self._generated_cache = None

        if is_eos is None:
            is_eos = int(token.cpu().item()) in self.eos_token_ids
        if is_eos or len(self._generated_tokens) >= self.max_new_tokens:
            self._state = Terminal("eos" if is_eos else "length", self.block_table)
        elif isinstance(state, Prefilling):
            self._state = DecodeReady(state.table)
