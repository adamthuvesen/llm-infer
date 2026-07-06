"""Preallocated device-side buffers for a planned decode window.

The classic ``decode_many`` rebuilds its bookkeeping from Python block tables every step:
per-layer ``prepare_write``/``physical_slot`` walks, a packed read plan re-listed from every
request's whole history, and fresh host-to-device copies for slots, positions, and RoPE rows.
At small-model batch sizes that Python work — not the GPU — is the decode wall.

A :class:`DecodeWindowPlan` does that bookkeeping once per window instead of once per step
(and the per-layer part not at all). At build time it reserves every block the window can
need, uploads the packed slot layout and the per-step write slots in one copy each, and
checks that no block is shared (copy-on-write can never trigger inside the window). Each
step then advances with a handful of device kernels, zero host-to-device traffic, and — key
for CUDA-graph capture — **zero host syncs**:

* ``begin_step`` scatters the step's write slots into the padded slot buffer and rebuilds the
  packed read indices from host-known sizes (every request grows by exactly one token per
  step, so the packed total is arithmetic, never a device-side count). The result is the
  exact request-major order ``PagedKVCache.plan_read_many`` produces, without the
  ``masked_select`` whose output-size query used to sync the host every step.
* ``complete_step`` advances positions, cumulative lengths, and the Python ``BlockTable``
  lengths so engine invariants (release, accounting) keep holding.

The plan is windowed, not global: the engine opens one per decode window over a stable
batch and drops it at the flush, so batch-composition changes never invalidate live buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.kv_cache.paged_kv_cache import KVPagePlan, KVReadPlan, PagedKVCache


@dataclass
class DecodeWindowPlan:
    """Reusable decode-step buffers for one window over a fixed request batch."""

    tables: list[BlockTable]
    # (budget, B): row ``s`` holds each request's physical write slot for window step ``s``.
    write_slot_matrix: torch.Tensor
    # (B, width): each request's physical slots for positions 0..len-1, zero-padded; grown by
    # one scattered column per step. ``width`` covers the whole window, so it never reallocates.
    read_slots: torch.Tensor
    # (B, page_width): each request's physical block ids, zero-padded. Feeds the native
    # paged-attention backend (FlashInfer, the default CUDA path); the packed gather fallback
    # leaves it untouched.
    page_indices: torch.Tensor
    # (B,): each request's next token position — the RoPE position and write column.
    positions: torch.Tensor
    # (B+1,) int32: packed-read cumulative lengths for the *upcoming* step.
    cu_seqlens: torch.Tensor
    # (B+1,) int32: per-step cu_seqlens increment (+1 token per request == +row index).
    cu_step: torch.Tensor
    # int64 twins of the two above — index arithmetic in the packed-read build needs long.
    cu_seqlens_long: torch.Tensor
    cu_step_long: torch.Tensor
    # (max_total,): flat packed positions for the window's largest step, sliced per step.
    flat_arange: torch.Tensor
    page_arange: torch.Tensor
    batch_arange: torch.Tensor  # (B,) — row indices for the per-step write-slot scatter
    base_lengths: list[int]  # host copy of each request's length at window open
    block_size: int
    max_len: int  # host-tracked max read length for the upcoming step (no device sync)
    total: int  # host-tracked packed size (sum of read lengths) for the upcoming step
    steps_used: int = 0

    @property
    def budget(self) -> int:
        return int(self.write_slot_matrix.shape[0])

    def begin_step(self, *, include_pages: bool = False) -> tuple[torch.Tensor, KVReadPlan]:
        """Write slots and packed read plan for the next decode step.

        Device-only work with host-known output shapes: no kernel here ever needs a
        device-to-host answer, so the step is sync-free and safe to run under
        ``torch.cuda.set_sync_debug_mode``.
        """
        if self.steps_used >= self.budget:
            raise ValueError(f"decode window exhausted: {self.steps_used}/{self.budget} steps")
        write_slots = self.write_slot_matrix[self.steps_used]
        self.read_slots[self.batch_arange, self.positions] = write_slots
        # Packed order is request-major: request i contributes its slots for positions
        # 0..len_i (inclusive of this step's write). Row/column of packed entry p are
        # ``rows[p] = i`` and ``cols[p] = p - cu_seqlens[i]`` — pure index arithmetic from
        # sizes the host already tracks, so ``output_size`` keeps repeat_interleave sync-free.
        rows = torch.repeat_interleave(
            self.batch_arange, self.positions + 1, output_size=self.total
        )
        cols = self.flat_arange[: self.total] - self.cu_seqlens_long.index_select(0, rows)
        lengths = [length + self.steps_used + 1 for length in self.base_lengths]
        read_plan = KVReadPlan(
            idx=self.read_slots[rows, cols],
            cu_seqlens=self.cu_seqlens,
            lengths=lengths,
            max_len=self.max_len,
            page_plan=self._page_plan(lengths) if include_pages else None,
        )
        return write_slots, read_plan

    def complete_step(self) -> None:
        """Advance to the next step: device counters plus the Python table lengths."""
        self.steps_used += 1
        self.positions += 1
        self.cu_seqlens += self.cu_step
        self.cu_seqlens_long += self.cu_step_long
        self.max_len += 1
        self.total += len(self.tables)
        for table in self.tables:
            table.length += 1

    def _page_plan(self, lengths: list[int]) -> KVPagePlan:
        """FlashInfer-style page metadata for the same request order as ``read_slots``."""
        page_counts = [-(-length // self.block_size) for length in lengths]
        indptr_host = [0]
        last_page_len_host: list[int] = []
        for length, page_count in zip(lengths, page_counts, strict=True):
            indptr_host.append(indptr_host[-1] + page_count)
            last_page_len_host.append(((length - 1) % self.block_size) + 1)

        device = self.page_indices.device
        indptr = torch.tensor(indptr_host, dtype=torch.int32).to(device, non_blocking=True)
        last_page_len = torch.tensor(last_page_len_host, dtype=torch.int32).to(
            device, non_blocking=True
        )
        counts = torch.tensor(page_counts, dtype=torch.long).to(device, non_blocking=True)
        total_pages = indptr_host[-1]
        rows = torch.repeat_interleave(self.batch_arange, counts, output_size=total_pages)
        cols = self.page_arange[:total_pages] - indptr.to(torch.long).index_select(0, rows)
        return KVPagePlan(
            indptr=indptr,
            indices=self.page_indices[rows, cols],
            last_page_len=last_page_len,
            page_size=self.block_size,
        )


def build_decode_window_plan(
    cache: PagedKVCache, tables: list[BlockTable], budget: int
) -> DecodeWindowPlan | None:
    """Build window buffers for ``tables``, or ``None`` when the batch is not plan-safe.

    Plan-safety means no request's blocks are shared (refcount 1 everywhere): the planned
    write path skips ``prepare_write``, so a copy-on-write append must be impossible. The
    engine already keeps prefix-group requests off the window path; this check makes the
    invariant loud rather than assumed. Reserves every block the window needs up front.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1; got {budget}")
    if not tables:
        raise ValueError("a decode window needs at least one table")
    for table in tables:
        if any(cache.allocator.refcount(block) > 1 for block in table.blocks):
            return None

    device = cache.key.device
    lengths = [table.length for table in tables]
    width = max(lengths) + budget
    page_width = max(-(-(length + budget) // cache.block_size) for length in lengths)

    write_rows: list[list[int]] = []
    history_rows: list[list[int]] = []
    page_rows: list[list[int]] = []
    for table, length in zip(tables, lengths, strict=True):
        table.reserve(budget)
        write_rows.append(table.physical_slots(length, budget))
        history_rows.append(table.physical_slots(0, length) + [0] * (width - length))
        page_count = -(-(length + budget) // cache.block_size)
        page_rows.append(table.blocks[:page_count] + [0] * (page_width - page_count))

    cu_host = [0]
    for length in lengths:
        cu_host.append(cu_host[-1] + length + 1)
    max_total = cu_host[-1] + (budget - 1) * len(tables)
    max_page_total = sum(-(-(length + budget) // cache.block_size) for length in lengths)

    def upload(values: list, dtype: torch.dtype) -> torch.Tensor:
        # Build on the host, then a non_blocking upload: `torch.tensor(..., device=cuda)`
        # ends in a blocking stream sync, which at window open would stall the CPU behind
        # the previous window's queued GPU work (the GPU probe flagged exactly these five
        # uploads). A pageable H2D copy is host-synchronous per CUDA semantics — the source
        # is staged before return — so the temporary's lifetime is safe without the sync.
        return torch.tensor(values, dtype=dtype).to(device, non_blocking=True)

    return DecodeWindowPlan(
        tables=list(tables),
        write_slot_matrix=upload(write_rows, torch.long).T.contiguous(),
        read_slots=upload(history_rows, torch.long),
        page_indices=upload(page_rows, torch.int32),
        positions=upload(lengths, torch.long),
        cu_seqlens=upload(cu_host, torch.int32),
        cu_step=torch.arange(len(tables) + 1, dtype=torch.int32, device=device),
        cu_seqlens_long=upload(cu_host, torch.long),
        cu_step_long=torch.arange(len(tables) + 1, dtype=torch.long, device=device),
        flat_arange=torch.arange(max_total, dtype=torch.long, device=device),
        page_arange=torch.arange(max_page_total, dtype=torch.long, device=device),
        batch_arange=torch.arange(len(tables), dtype=torch.long, device=device),
        base_lengths=lengths,
        block_size=cache.block_size,
        max_len=max(lengths) + 1,
        total=cu_host[-1],
    )
