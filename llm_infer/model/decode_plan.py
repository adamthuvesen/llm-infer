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

* ``begin_step`` returns the step's precomputed write slots. Packed-attention fallbacks scatter
  them into the padded history and rebuild their read indices from host-known sizes; native
  paged-attention backends skip that work and update stable, preallocated page metadata buffers.
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
    # Packed fallback only. (B, width): each request's physical slots for positions 0..len-1,
    # zero-padded and grown by one scattered column per step.
    read_slots: torch.Tensor | None
    # (B,): each request's next token position — the RoPE position and write column.
    positions: torch.Tensor
    # (B+1,) int32: packed-read cumulative lengths for the *upcoming* step.
    cu_seqlens: torch.Tensor | None
    # (B+1,) int32: per-step cu_seqlens increment (+1 token per request == +row index).
    cu_step: torch.Tensor | None
    # int64 twins of the two above — index arithmetic in the packed-read build needs long.
    cu_seqlens_long: torch.Tensor | None
    cu_step_long: torch.Tensor | None
    # (max_total,): flat packed positions for the window's largest step, sliced per step.
    flat_arange: torch.Tensor | None
    # Native-page metadata for every possible step, copied into the stable buffers below.
    # Keeping the destination addresses fixed is the prerequisite for capturing attention.
    page_indptr_matrix: torch.Tensor | None
    page_indices_matrix: torch.Tensor | None
    page_last_page_len_matrix: torch.Tensor | None
    page_totals: list[int] | None
    page_indptr: torch.Tensor | None
    native_page_indices: torch.Tensor | None
    page_last_page_len: torch.Tensor | None
    batch_arange: torch.Tensor  # (B,) — row indices for the per-step write-slot scatter
    base_lengths: list[int]  # host copy of each request's length at window open
    block_size: int
    max_len: int  # host-tracked max read length for the upcoming step (no device sync)
    total: int  # host-tracked packed size (sum of read lengths) for the upcoming step
    steps_used: int = 0

    @property
    def budget(self) -> int:
        return int(self.write_slot_matrix.shape[0])

    def begin_step(
        self, *, include_pages: bool = False, include_packed: bool = True
    ) -> tuple[torch.Tensor, KVReadPlan]:
        """Write slots and the requested read metadata for the next decode step.

        Device-only work with host-known output shapes: no kernel here ever needs a
        device-to-host answer, so the step is sync-free and safe to run under
        ``torch.cuda.set_sync_debug_mode``.
        """
        if self.steps_used >= self.budget:
            raise ValueError(f"decode window exhausted: {self.steps_used}/{self.budget} steps")
        if not include_pages and not include_packed:
            raise ValueError("a decode step needs packed indices or native page metadata")
        write_slots = self.write_slot_matrix[self.steps_used]
        # Packed order is request-major: request i contributes its slots for positions
        # 0..len_i (inclusive of this step's write). Row/column of packed entry p are
        # ``rows[p] = i`` and ``cols[p] = p - cu_seqlens[i]`` — pure index arithmetic from
        # sizes the host already tracks, so ``output_size`` keeps repeat_interleave sync-free.
        if include_packed:
            if (
                self.read_slots is None
                or self.cu_seqlens is None
                or self.cu_seqlens_long is None
                or self.flat_arange is None
            ):
                raise ValueError("decode window was built without packed read buffers")
            self.read_slots[self.batch_arange, self.positions] = write_slots
            rows = torch.repeat_interleave(
                self.batch_arange, self.positions + 1, output_size=self.total
            )
            cols = self.flat_arange[: self.total] - self.cu_seqlens_long.index_select(0, rows)
            packed_idx = self.read_slots[rows, cols]
        else:
            packed_idx = None
        lengths = [length + self.steps_used + 1 for length in self.base_lengths]
        read_plan = KVReadPlan(
            idx=packed_idx,
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
        if self.cu_seqlens is not None and self.cu_step is not None:
            self.cu_seqlens += self.cu_step
        if self.cu_seqlens_long is not None and self.cu_step_long is not None:
            self.cu_seqlens_long += self.cu_step_long
        self.max_len += 1
        self.total += len(self.tables)
        for table in self.tables:
            table.length += 1

    def _page_plan(self, lengths: list[int]) -> KVPagePlan:
        """Update and return the production fixed-address page buffers."""
        if (
            self.page_indptr_matrix is None
            or self.page_indices_matrix is None
            or self.page_last_page_len_matrix is None
            or self.page_totals is None
            or self.page_indptr is None
            or self.native_page_indices is None
            or self.page_last_page_len is None
        ):
            raise ValueError("decode window was built without native page metadata")
        step = self.steps_used
        total_pages = self.page_totals[step]
        self.page_indptr.copy_(self.page_indptr_matrix[step])
        self.native_page_indices[:total_pages].copy_(self.page_indices_matrix[step, :total_pages])
        self.page_last_page_len.copy_(self.page_last_page_len_matrix[step])
        return KVPagePlan(
            indptr=self.page_indptr,
            indices=self.native_page_indices[:total_pages],
            last_page_len=self.page_last_page_len,
            page_size=self.block_size,
        )


def build_decode_window_plan(
    cache: PagedKVCache,
    tables: list[BlockTable],
    budget: int,
    *,
    include_pages: bool = False,
    include_packed: bool = True,
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
    if not include_pages and not include_packed:
        raise ValueError("a decode window needs packed indices or native page metadata")
    for table in tables:
        if any(cache.allocator.refcount(block) > 1 for block in table.blocks):
            return None

    device = cache.kv.device
    lengths = [table.length for table in tables]
    width = max(lengths) + budget
    write_rows: list[list[int]] = []
    history_rows: list[list[int]] | None = [] if include_packed else None
    for table, length in zip(tables, lengths, strict=True):
        table.reserve(budget)
        write_rows.append(table.physical_slots(length, budget))
        if history_rows is not None:
            history_rows.append(table.physical_slots(0, length) + [0] * (width - length))

    cu_host = [0]
    for length in lengths:
        cu_host.append(cu_host[-1] + length + 1)
    max_total = cu_host[-1] + (budget - 1) * len(tables) if include_packed else 0
    page_indptr_rows: list[list[int]] | None = None
    page_index_rows: list[list[int]] | None = None
    page_last_rows: list[list[int]] | None = None
    page_totals: list[int] | None = None
    max_page_total = 0
    if include_pages:
        page_indptr_rows = []
        page_index_rows = []
        page_last_rows = []
        page_totals = []
        for step in range(budget):
            step_lengths = [length + step + 1 for length in lengths]
            counts = [-(-length // cache.block_size) for length in step_lengths]
            indptr = [0]
            indices: list[int] = []
            last_page_lengths: list[int] = []
            for table, length, count in zip(tables, step_lengths, counts, strict=True):
                indptr.append(indptr[-1] + count)
                indices.extend(table.blocks[:count])
                last_page_lengths.append(((length - 1) % cache.block_size) + 1)
            page_indptr_rows.append(indptr)
            page_index_rows.append(indices)
            page_last_rows.append(last_page_lengths)
            page_totals.append(indptr[-1])
        max_page_total = max(page_totals)
        for row in page_index_rows:
            row.extend([0] * (max_page_total - len(row)))

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
        read_slots=upload(history_rows, torch.long) if history_rows is not None else None,
        positions=upload(lengths, torch.long),
        cu_seqlens=upload(cu_host, torch.int32) if include_packed else None,
        cu_step=(
            torch.arange(len(tables) + 1, dtype=torch.int32, device=device)
            if include_packed
            else None
        ),
        cu_seqlens_long=upload(cu_host, torch.long) if include_packed else None,
        cu_step_long=(
            torch.arange(len(tables) + 1, dtype=torch.long, device=device)
            if include_packed
            else None
        ),
        flat_arange=(
            torch.arange(max_total, dtype=torch.long, device=device) if include_packed else None
        ),
        page_indptr_matrix=(
            upload(page_indptr_rows, torch.int32) if page_indptr_rows is not None else None
        ),
        page_indices_matrix=(
            upload(page_index_rows, torch.int32) if page_index_rows is not None else None
        ),
        page_last_page_len_matrix=(
            upload(page_last_rows, torch.int32) if page_last_rows is not None else None
        ),
        page_totals=page_totals,
        page_indptr=(
            torch.empty(len(tables) + 1, dtype=torch.int32, device=device)
            if include_pages
            else None
        ),
        native_page_indices=(
            torch.empty(max_page_total, dtype=torch.int32, device=device) if include_pages else None
        ),
        page_last_page_len=(
            torch.empty(len(tables), dtype=torch.int32, device=device) if include_pages else None
        ),
        batch_arange=torch.arange(len(tables), dtype=torch.long, device=device),
        base_lengths=lengths,
        block_size=cache.block_size,
        max_len=max(lengths) + 1,
        total=cu_host[-1],
    )
