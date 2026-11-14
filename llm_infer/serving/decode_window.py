"""Deferred decode buffers and their ordered device-to-host transfer lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from llm_infer.model.decode_plan import DecodeWindowPlan
from llm_infer.profiling import TimingProfiler
from llm_infer.request.request import Request


@dataclass
class DecodeWindow:
    """Deferred-decode state for one stable batch of greedy or penalty-free sampled rows.

    While a window is open, each decode step's sampled tokens stay on device in ``pending``
    (one ``(B,)`` long tensor per step) and no EOS/stop host sync happens. The flush *stages*
    the window instead of syncing: one ``non_blocking`` device-to-host copy of (tokens, EOS
    flags) into pinned buffers, consumed one window later (see :class:`PendingWindowFlush`).
    A request that hit EOS mid-window therefore decodes throwaway tokens — bounded by the
    current window's remainder plus one whole follow-up window — which are discarded at
    consume time, never emitted. That is waste, not a correctness change: recorded tokens
    stay identical to the per-step engine. A sampled row's overshoot steps also consume
    draws from its own generator; those draws die with the discarded tokens and no other
    request shares that generator, so replay of the recorded tokens is unaffected.
    """

    request_ids: tuple[str, ...]
    requests: list[Request]
    # How many steps this window may run: the sync interval, capped by the batch's smallest
    # remaining token budget (minus tokens still staged in an unconsumed flush) so no
    # request can decode past its own max_new_tokens.
    budget: int
    # Computed once at open: an all-greedy window's per-step selection stays the bare batched
    # argmax, adding no per-step Python to the proven greedy path.
    all_greedy: bool = True
    pending: list[torch.Tensor] = field(default_factory=list)
    # Preallocated model-side step buffers (planned_decode backends); None falls back to
    # per-step ``decode_many`` bookkeeping inside the same window.
    plan: DecodeWindowPlan | None = None


@dataclass
class PendingWindowFlush:
    """A flushed window's tokens, in flight to the host, consumed one window behind.

    Staging instead of syncing keeps the flush off the step loop's critical path: the
    device-to-host copy overlaps the *next* window's GPU work, and by the time that window
    flushes, ``event`` has long signalled — the consume reads pinned memory without waiting
    on the compute stream. The device ``matrix`` stays referenced here so the recorded
    per-token views (and the next window's first input, ``matrix[-1]``) remain valid.
    """

    request_ids: tuple[str, ...]
    requests: list[Request]
    matrix: torch.Tensor  # (S, B) on the model device — record views + next window's input
    tokens_host: torch.Tensor  # (S, B) on the host (pinned when the model is on CUDA)
    # (S, B) bool EOS flags computed on device (single shared stop set), or None when the
    # batch mixes stop sets — the consume then checks membership host-side per token.
    eos_host: torch.Tensor | None
    steps: int
    event: torch.cuda.Event | None  # None on CPU, where the copy is already synchronous


class DecodeWindows:
    """Own active/staged tensors until the engine consumes their recorded tokens.

    Request mutation and release stay with the engine. This object only stages buffers
    and returns completed stages in decode order. A stage retains its device matrix,
    including views used as the next window's input, until consumption finishes.
    """

    def __init__(self, *, profiler: TimingProfiler | None = None) -> None:
        self.active: DecodeWindow | None = None
        self.pending: PendingWindowFlush | None = None
        self._eos_tensors: dict[frozenset[int], torch.Tensor] = {}
        self._profiler = profiler

    def flush(self) -> PendingWindowFlush | None:
        """Stage the active window and return the previous stage, if it became consumable."""
        window = self.active
        self.active = None
        previous = self.pending
        if window is not None and window.pending:
            if self._profiler is None:
                self.pending = self._stage(window)
            else:
                with self._profiler.host("window_flushing"):
                    self.pending = self._stage(window)
        return previous if previous is not self.pending else None

    def drain(self) -> list[PendingWindowFlush]:
        """Detach all staged copies in decode order for synchronous engine consumption."""
        previous = self.flush()
        pending = self.pending
        self.pending = None
        return [stage for stage in (previous, pending) if stage is not None]

    def _stage(self, window: DecodeWindow) -> PendingWindowFlush:
        """Enqueue the window's device-to-host copy (pinned, non-blocking) without waiting."""
        matrix = torch.stack(window.pending)  # (S, B), on the model device
        eos_mask: torch.Tensor | None = None
        eos_sets = {request.eos_token_ids for request in window.requests}
        if len(eos_sets) == 1:
            lookup = self._eos_lookup(next(iter(eos_sets)), matrix.device)
            if lookup.numel():
                eos_mask = (matrix.unsqueeze(-1) == lookup).any(dim=-1)
        if matrix.is_cuda:
            tokens_host = torch.empty(matrix.shape, dtype=matrix.dtype, pin_memory=True)
            tokens_host.copy_(matrix, non_blocking=True)
            eos_host: torch.Tensor | None = None
            if eos_mask is not None:
                eos_host = torch.empty(eos_mask.shape, dtype=torch.bool, pin_memory=True)
                eos_host.copy_(eos_mask, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        else:
            tokens_host = matrix
            eos_host = eos_mask
            event = None
        return PendingWindowFlush(
            request_ids=window.request_ids,
            requests=window.requests,
            matrix=matrix,
            tokens_host=tokens_host,
            eos_host=eos_host,
            steps=len(window.pending),
            event=event,
        )

    def _eos_lookup(self, eos_token_ids: frozenset[int], device: torch.device) -> torch.Tensor:
        """The EOS-id comparison tensor for one stop set, built once per engine and reused.

        Rebuilding this tensor every decode step is a host-to-device copy in the hot loop; the
        stop sets in play are tiny and stable, so a per-engine cache removes it. The engine's
        device is fixed for its lifetime, so the set alone is a sufficient key.
        """
        cached = self._eos_tensors.get(eos_token_ids)
        if cached is None:
            cached = torch.tensor(sorted(eos_token_ids), dtype=torch.long, device=device)
            self._eos_tensors[eos_token_ids] = cached
        return cached
