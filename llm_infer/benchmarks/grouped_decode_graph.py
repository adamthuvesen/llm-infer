"""Benchmark-only exact-batch graph grouping complete Esme layers with paged attention."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from llm_infer.benchmarks.flashinfer_graph_probe import (
    FlashInferPageMetadata,
    FlashInferProbeShape,
    build_graph_wrapper,
    plan_wrapper,
    run_wrapper_into,
)
from llm_infer.model.decode_graph import DecodeGraphRunner

if TYPE_CHECKING:
    from llm_infer.kv_cache.paged_kv_cache import KVReadPlan, PagedKVCache
    from llm_infer.model.decode_plan import DecodeWindowPlan
    from llm_infer.model.pretrain_bundle import PretrainBundleModel


class EngineOwnedGroupedDecodeGraphRunner:
    """Tie one exact-batch grouped-layer graph to one engine cache.

    ``mode="eager"`` runs the same tranche ordering on CPU for parity tests. Graph mode
    captures segments ``0..grouped_layers`` around their KV writes and a dedicated
    FlashInfer graph-mode wrapper. Planning remains outside capture on every token step.
    """

    def __init__(
        self,
        model: PretrainBundleModel,
        cache: PagedKVCache,
        batch_size: int,
        *,
        grouped_layers: int,
        mode: str,
        capture_plan: DecodeWindowPlan | None = None,
        max_position: int = 8192,
    ) -> None:
        if mode not in ("eager", "graph"):
            raise ValueError(f"mode must be 'eager' or 'graph'; got {mode!r}")
        if batch_size not in (1, 8):
            raise ValueError(f"grouped probe supports exact batches 1 and 8; got {batch_size}")
        if grouped_layers not in (2, 4):
            raise ValueError(f"grouped_layers must be 2 or 4; got {grouped_layers}")
        if model.num_layers < grouped_layers:
            raise ValueError(
                f"grouped probe needs {grouped_layers} layers; got {model.num_layers}"
            )
        if mode == "graph" and capture_plan is None:
            raise ValueError("graph mode needs a prefilled capture plan")
        if mode == "graph" and model.num_layers == grouped_layers:
            raise ValueError("graph mode needs a following layer after the grouped tranche")
        self.model = model
        self.cache = cache
        self.batch_size = batch_size
        self.grouped_layers = grouped_layers
        self.mode = mode
        self.requires_packed_read_plan = not model._uses_paged_decode_backend()
        self.piecewise = DecodeGraphRunner(
            model,
            capture_sizes=(batch_size,),
            max_position=max_position,
            mode=mode,
        )
        self._state = self.piecewise._buckets[batch_size]
        self._active_plan: DecodeWindowPlan | None = None
        self._cache_kv_pointer = cache.kv.data_ptr()
        self._static_write_slots = torch.zeros(
            batch_size, dtype=torch.long, device=model.device
        )
        self._group_graph: torch.cuda.CUDAGraph | None = None
        self._graph_wrapper = None
        self._graph_shape: FlashInferProbeShape | None = None
        self._graph_workspace: torch.Tensor | None = None
        self._fixed_metadata: FlashInferPageMetadata | None = None
        self.capture_seconds = 0.0
        self.capture_memory_bytes = 0
        self.owned_group_memory_bytes = 0
        self.total_capture_seconds = 0.0
        if mode == "graph":
            assert capture_plan is not None
            self._capture_group(capture_plan)

    def window_step(
        self, cache: PagedKVCache, plan: DecodeWindowPlan, token_ids: torch.Tensor
    ) -> torch.Tensor | None:
        """Run one exact-batch step with the configured leading layers grouped."""
        if len(plan.tables) != self.batch_size:
            return None
        if cache is not self.cache or cache.kv.data_ptr() != self._cache_kv_pointer:
            return None
        state = self._state
        if plan.steps_used == 0:
            if max(plan.base_lengths) + plan.budget > self.piecewise.max_position:
                return None
            state.positions.copy_(plan.positions)
            self._active_plan = plan
        elif self._active_plan is not plan:
            return None

        uses_native_pages = self.model._uses_paged_decode_backend()
        write_slots, read_plan = plan.begin_step(
            include_pages=uses_native_pages,
            include_packed=not uses_native_pages,
        )
        state.tokens.copy_(token_ids.reshape(-1))
        self._static_write_slots.copy_(write_slots)
        if self._group_graph is None:
            self.model._prepare_paged_decode(read_plan)
            self._run_group_eager(read_plan)
            for layer in range(self.grouped_layers, self.model.num_layers):
                self.piecewise._attention_eager(
                    state,
                    layer,
                    cache,
                    self.batch_size,
                    self._static_write_slots,
                    read_plan,
                )
                self.piecewise._run_segment(state, layer + 1)
        else:
            self._plan_group_wrapper(read_plan)
            self._group_graph.replay()
            for layer in range(self.grouped_layers, self.model.num_layers):
                cache.write_rows(
                    layer,
                    self._static_write_slots,
                    state.k,
                    state.v,
                )
                run_wrapper_into(
                    self._graph_wrapper,
                    state.q,
                    cache.layer_kv(layer),
                    state.attn,
                )
                self.piecewise._run_segment(state, layer + 1)
        plan.complete_step()
        return state.logits

    def _run_group_eager(self, read_plan: KVReadPlan) -> None:
        state = self._state
        self.piecewise._run_segment(state, 0)
        for layer in range(self.grouped_layers):
            self.piecewise._attention_eager(
                state,
                layer,
                self.cache,
                self.batch_size,
                self._static_write_slots,
                read_plan,
            )
            self.piecewise._run_segment(state, layer + 1)

    def _run_group_graph_body(self) -> None:
        if self._graph_wrapper is None:
            raise RuntimeError("FlashInfer graph wrapper is missing")
        state = self._state
        state.run_segments[0]()
        for layer in range(self.grouped_layers):
            self.cache.write_rows(
                layer,
                self._static_write_slots,
                state.k,
                state.v,
            )
            run_wrapper_into(
                self._graph_wrapper,
                state.q,
                self.cache.layer_kv(layer),
                state.attn,
            )
            state.run_segments[layer + 1]()

    def _capture_group(self, capture_plan: DecodeWindowPlan) -> None:
        backend = self.model._paged_backend
        if backend is None:
            raise ValueError("graph mode needs a native paged attention backend")
        write_slots, read_plan = capture_plan.begin_step(
            include_pages=True, include_packed=False
        )
        if read_plan.page_plan is None:
            raise RuntimeError("capture plan did not produce native page metadata")
        self._static_write_slots.copy_(write_slots)
        self._state.positions.copy_(capture_plan.positions)
        fixed_metadata = FlashInferPageMetadata(
            indptr=torch.empty(
                self.batch_size + 1, dtype=torch.int32, device=self.model.device
            ),
            indices=torch.empty(
                self.cache.num_blocks, dtype=torch.int32, device=self.model.device
            ),
            last_page_len=torch.empty(
                self.batch_size, dtype=torch.int32, device=self.model.device
            ),
        )
        # FlashInfer requires a zeroed workspace before the wrapper's first use.
        workspace = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.model.device
        )
        wrapper_class = backend._decode_wrapper_class(backend._flashinfer)
        self._graph_wrapper = build_graph_wrapper(wrapper_class, workspace, fixed_metadata)
        self._graph_workspace = workspace
        self._fixed_metadata = fixed_metadata
        self._graph_shape = FlashInferProbeShape(
            self.model.num_heads,
            self.model.num_kv_heads,
            self.model.head_dim,
            read_plan.page_plan.page_size,
            self.model.dtype,
        )
        self._plan_group_wrapper(read_plan)
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        start = time.perf_counter()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._run_group_graph_body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_group_graph_body()
        torch.cuda.synchronize()
        self._group_graph = graph
        self.capture_seconds = time.perf_counter() - start
        self.capture_memory_bytes = torch.cuda.memory_allocated() - allocated_before
        fixed_buffer_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                workspace,
                fixed_metadata.indptr,
                fixed_metadata.indices,
                fixed_metadata.last_page_len,
                self._static_write_slots,
            )
        )
        self.owned_group_memory_bytes = fixed_buffer_bytes + self.capture_memory_bytes
        # Capture writes the upcoming slots with warmup values. Every real layer overwrites
        # its own slot before attending to it, so no captured value reaches generation.

    def _plan_group_wrapper(self, read_plan: KVReadPlan) -> None:
        if (
            self._graph_wrapper is None
            or self._graph_shape is None
            or read_plan.page_plan is None
        ):
            raise RuntimeError("grouped graph wrapper cannot plan this read metadata")
        plan_wrapper(
            self._graph_wrapper,
            FlashInferPageMetadata(
                indptr=read_plan.page_plan.indptr,
                indices=read_plan.page_plan.indices,
                last_page_len=read_plan.page_plan.last_page_len,
            ),
            self._graph_shape,
        )
