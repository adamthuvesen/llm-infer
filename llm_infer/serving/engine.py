"""Continuous-batching decode loop: one step advances decode and prefill work."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

import torch

from llm_infer.kv_cache.block_allocator import BlockPoolEvent, OutOfBlocksError
from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.kv_cache.prefix_cache import PrefixCacheStore
from llm_infer.model.decode_plan import DecodeWindowPlan
from llm_infer.model.grouped_decode_graph import (
    DEFAULT_GROUPED_CAPTURE_SIZES,
    EngineOwnedGroupedDecodeGraphRunner,
)
from llm_infer.model.interface import (
    BackendCapabilities,
    BatchedPrefillBackend,
    CausalLMBackend,
    PlannedDecodeBackend,
)
from llm_infer.model.runtime import prepare_engine_grouped_decode, resolve_backend_capabilities
from llm_infer.profiling import TimingProfiler
from llm_infer.scheduler.scheduler import Scheduler, blocks_for_footprint, max_blocks_for
from llm_infer.serving.decode_window import DecodeWindow, DecodeWindows, PendingWindowFlush
from llm_infer.serving.request import DecodeReady, Pending, Prefilling, Request
from llm_infer.serving.sampler import GREEDY, SamplingParams, sample_row, sample_rows
from llm_infer.serving.speculative import PromptLookupDraft, SpeculativeDecodingConfig
from llm_infer.tracing import (
    FinishReason,
    PreemptReason,
    TokenSource,
    TraceEvent,
    TraceEventName,
    TraceRecorder,
)


@dataclass
class StepResult:
    """What happened in one engine step — enough to trace the vertical slice."""

    admitted: list[str] = field(default_factory=list)
    prefill_chunks: dict[str, tuple[int, int]] = field(default_factory=dict)
    finished: list[str] = field(default_factory=list)
    tokens: dict[str, list[int | torch.Tensor]] = field(default_factory=dict)
    finished_outputs: dict[str, list[int]] = field(default_factory=dict)


class InferenceEngine:
    """Runs generation for a set of requests over a shared paged KV-cache (greedy by default)."""

    def __init__(
        self,
        model: CausalLMBackend,
        *,
        block_size: int,
        num_blocks: int,
        device: str = "cpu",
        default_sampling: SamplingParams | None = None,
        profiler: TimingProfiler | None = None,
        prefill_chunk_size: int | None = None,
        speculative: SpeculativeDecodingConfig | None = None,
        preemption: bool = False,
        trace: TraceRecorder | None = None,
        trace_clock: Callable[[], float] | None = None,
        capabilities: BackendCapabilities | None = None,
        decode_window_size: int = 8,
        prefix_cache: bool = False,
        batched_prefill: bool = True,
        grouped_decode_graphs: bool = False,
        grouped_capture_sizes: tuple[int, ...] = DEFAULT_GROUPED_CAPTURE_SIZES,
        grouped_layers: int | None = None,
        grouped_shared_workspace: bool = True,
    ) -> None:
        if prefill_chunk_size is not None and prefill_chunk_size < 1:
            raise ValueError(f"prefill_chunk_size must be >= 1 when set; got {prefill_chunk_size}")
        if decode_window_size < 1:
            raise ValueError(f"decode_window_size must be >= 1; got {decode_window_size}")
        self.model = model
        self.capabilities = (
            capabilities if capabilities is not None else resolve_backend_capabilities(model)
        )
        self._planned_backend = model if isinstance(model, PlannedDecodeBackend) else None
        if self.capabilities.planned_decode and self._planned_backend is None:
            raise ValueError(
                "planned_decode capability requires open_decode_window and decode_window_step"
            )
        if speculative is not None and not self.capabilities.speculative:
            raise ValueError(
                "speculative decoding is not supported by this backend; "
                "omit speculative= or use a paged-KV backend"
            )
        if preemption and not self.capabilities.paged_kv:
            raise ValueError(
                "preemption requires a paged-KV backend; omit preemption=True for a backend "
                "that does not advertise paged_kv"
            )
        self.cache = PagedKVCache(
            num_layers=model.num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=model.num_kv_heads,
            head_dim=model.head_dim,
            dtype=model.dtype,
            device=device,
        )
        self.preemption = preemption
        self.scheduler = Scheduler(num_blocks, block_size, preemption=preemption)
        # Cross-turn prefix cache: donated prompt-prefix blocks survive one request's finish so a
        # follow-up turn reuses them instead of re-prefilling the whole conversation. It over-holds
        # blocks outside the free pool and gives them back via the allocator's eviction hook, which
        # is incompatible with preemption's real-free-block footprint admission — reject the combo.
        self.prefix_cache: PrefixCacheStore | None = None
        if prefix_cache:
            if preemption:
                raise ValueError(
                    "prefix_cache and preemption cannot be enabled together; the preemption "
                    "scheduler admits against real free blocks and would fight the cache"
                )
            if not (self.capabilities.prefix_caching and self.capabilities.paged_kv):
                raise ValueError(
                    "prefix_cache requires a paged-KV backend that advertises prefix caching"
                )
            self.prefix_cache = PrefixCacheStore(self.cache.allocator, block_size)
        # The fallback sampling for a request that carries no params of its own. Defaults to
        # greedy (temperature 0, token-for-token the checked reference path).
        self.default_sampling = default_sampling or GREEDY
        self.profiler = profiler
        self.model.profiler = profiler
        self.prefill_chunk_size = prefill_chunk_size
        self.batched_prefill = batched_prefill
        self.speculative = PromptLookupDraft(speculative) if speculative is not None else None
        self.trace = trace
        if trace is not None:
            # Emit block_allocated/block_freed from the real physical boundary: the allocator
            # is the single owner of the free pool, so its events are clear for refcounted
            # prefix sharing and copy-on-write. Nothing is faked from request state.
            self.cache.allocator.observer = self._trace_pool_event
        self._trace_clock = trace_clock or time.perf_counter
        self._step_index = 0
        self._trace_sequence = 0
        self._trace_step: int | None = None
        self._trace_start_time = self._trace_clock()
        self._trace_total_tokens = 0
        self._last_traced_batch_size = 0
        self._requests: dict[str, Request] = {}
        # Deferred-decode window state: how many one-token decode steps may run between EOS/stop
        # host boundaries (1 = classic per-step sync), the currently open window, and the
        # staged flush of the previous window (consumed one window behind; see
        # PendingWindowFlush).
        self.decode_window_size = decode_window_size
        self._decode_windows = DecodeWindows(profiler=profiler)
        # A window flush forced by abort() lands here and is merged into the next step's
        # result, so other requests' flushed tokens still reach the serving loop.
        self._stashed_flush: StepResult | None = None
        # Observability only: a running tally of preemptions for the serving metrics endpoint.
        # Read live at scrape time; it never influences scheduling or decoding.
        self.preemption_count = 0
        # Engine-owned grouped decode graphs, one runner per exact batch size. Unlike the
        # model-owned piecewise runner (cache-agnostic; KV writes stay eager), grouped
        # captures bake this engine's cache KV pointer into the graphs, so the engine — the
        # cache owner — constructs them. Captured here, at construction, so no capture cost
        # can land inside a timed or serving region.
        self.grouped_decode_runners: dict[int, EngineOwnedGroupedDecodeGraphRunner] = {}
        if grouped_decode_graphs:
            self.grouped_decode_runners = prepare_engine_grouped_decode(
                model,
                self.cache,
                self.capabilities,
                capture_sizes=grouped_capture_sizes,
                grouped_layers=grouped_layers,
                shared_workspace=grouped_shared_workspace,
            )

    @property
    def grouped_decode_steps_total(self) -> int:
        """Window steps the grouped runners actually ran — the silent-fallback tripwire.

        Exposed on ``/metrics`` so a deployment where the grouped path quietly never fires
        (pointer mismatch, off-bucket batches) is visible, not just slower.
        """
        return sum(runner.steps_handled for runner in self.grouped_decode_runners.values())

    @property
    def prefix_cache_hit_tokens_total(self) -> int:
        """KV positions served from the prefix cache instead of being re-prefilled (0 when off)."""
        return self.prefix_cache.hit_tokens if self.prefix_cache is not None else 0

    @property
    def prefix_cache_hits_total(self) -> int:
        """Prefill lookups that reused a block-aligned prefix (0 when the cache is off)."""
        return self.prefix_cache.hits if self.prefix_cache is not None else 0

    @property
    def prefix_cache_misses_total(self) -> int:
        """Prefill lookups that found no reusable prefix (0 when the cache is off)."""
        return self.prefix_cache.misses if self.prefix_cache is not None else 0

    def add_request(self, request: Request) -> None:
        """Register and queue a request. Duplicate ids are rejected loudly."""
        if request.state != Pending() or request.generated_count:
            raise ValueError("only an unstarted request can be added to an engine")
        if request.prefix_group_id is not None and not self.capabilities.prefix_caching:
            raise ValueError(
                "prefix_group_id requires prefix caching; this backend does not support it"
            )
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request id {request.request_id!r}")
        self.scheduler.add(request)
        self._requests[request.request_id] = request

    def abort(self, request_id: str) -> bool:
        """Drop a request mid-flight, freeing its KV — for a client that disconnected.

        Handles both scheduler states clearly: a still-waiting request is removed from the
        queue (no KV yet), a running request has its block table freed at the allocator
        boundary and its budget released, exactly like the finish-sweep. Idempotent: a request
        already finished or unknown returns ``False``. Must be called between steps (the
        serving loop owns the engine on one thread), never mid-forward.

        An open deferred-decode window (and any staged flush pipelined behind it) is drained
        first so every request's recorded tokens are current before any window member is
        dropped. The drained tokens belong to *other* still-streaming requests too, so they
        are stashed and merged into the next ``step()`` result rather than discarded.
        """
        if self._decode_windows.active is not None or self._decode_windows.pending is not None:
            stashed = self._stashed_flush or StepResult()
            self._drain_decode_window(stashed)
            self._stashed_flush = stashed
        request = self._requests.pop(request_id, None)
        if request is None:
            return False
        if self.scheduler.is_running(request) and request.block_table is not None:
            self._free_block_table(request.block_table)
        request.cancel()
        request.release_finished_table()
        return self.scheduler.cancel(request)

    def step(self) -> StepResult:
        """Admit, advance every running request by one token, then free finished ones.

        Newly-admitted requests cache at most ``prefill_chunk_size`` prompt tokens, or their
        full prompt when no chunk limit is configured. A request becomes decode-ready only
        when its full prompt has been cached and its first token sampled. Every already-ready
        request advances by one token through a single **batched** decode forward
        (``decode_many``) rather than one forward each — the fused-batch decode that makes
        continuous batching a throughput win, not just a scheduling one.
        """
        result = self._stashed_flush or StepResult()
        self._stashed_flush = None
        self._trace_step = self._step_index
        self._step_index += 1
        try:
            admit_kwargs = {"free_blocks": self.cache.allocator.num_free} if self.preemption else {}
            for request in self.scheduler.admit(**admit_kwargs):
                result.admitted.append(request.request_id)
                self._emit_trace(
                    "request_admitted",
                    request_id=request.request_id,
                    prompt_tokens=len(request.prompt_ids),
                    max_new_tokens=request.max_new_tokens,
                    prefix_group_id=request.prefix_group_id,
                    reserved_blocks=self._reserved_blocks(request),
                )
            self._trace_batch_size_changed()

            to_decode: list[Request] = []
            to_prefill: list[Request] = []
            to_resume: list[Request] = []
            for request in self.scheduler.running:
                state = request.state
                if isinstance(state, DecodeReady):
                    to_decode.append(request)
                elif isinstance(state, (Pending, Prefilling)) and state.mode == "recompute":
                    # Preempted earlier: rebuild its KV by recompute before it decodes again.
                    to_resume.append(request)
                else:
                    to_prefill.append(request)

            self._resume_requests(to_resume, result)
            self._prefill_requests(to_prefill, result)

            if to_decode:
                self._decode_requests(to_decode, result)
                # Multi-step scheduling: with a deferred window open and nothing waiting to
                # admit, run the window's remaining steps in this same scheduler pass — the
                # admission/classification bookkeeping above is per pass, not per token. A
                # non-empty waiting queue keeps the classic one-step-per-pass cadence so a
                # mid-window finisher can free blocks for admission at the next pass.
                while self._decode_windows.active is not None and not self.scheduler.waiting:
                    self._decode_window_step(self._decode_windows.active.requests, result)

            # Finishers are freed inside each prefill/decode op (see _release_finished_in), so by
            # here the running set already excludes them; nothing left to sweep.
            self._trace_batch_size_changed()
            self._trace_throughput_sample(result)

            return result
        finally:
            self._trace_step = None

    def run(self) -> dict[str, list[int]]:
        """Step until the queue and running set drain; return each request's generated ids."""
        finished_outputs: dict[str, list[int]] = {}
        with self._record_time("total_wall"):
            while self.scheduler.has_work():
                result = self.step()
                finished_outputs.update(result.finished_outputs)
        return finished_outputs

    def _reserved_blocks(self, request: Request) -> int:
        """Blocks the admit event reports reserved — footprint under preemption, else worst case."""
        if self.preemption:
            return blocks_for_footprint(request, self.scheduler.block_size)
        return max_blocks_for(request, self.scheduler.block_size)

    def _prefill_requests(self, requests: list[Request], result: StepResult) -> None:
        """Prefill unstarted requests, sharing prompt blocks for declared sibling groups."""
        if self.prefix_cache is not None:
            for request in requests:
                self._apply_prefix_cache_hit(request)
        if self._can_prefill_many(requests):
            self._prefill_many(requests, result)
            return

        handled: set[str] = set()
        for request in requests:
            if request.request_id in handled:
                continue
            # A request can be preempted out of the running set while we make pool room for an
            # earlier one this step; check membership live (see _is_running) so it is skipped and
            # re-prefills once re-admitted, never prefilled while sitting in the waiting queue.
            if self.preemption and not self._is_running(request):
                continue
            if request.prefix_group_id is None:
                self._prefill_one(request, result)
                handled.add(request.request_id)
                continue

            group = [
                candidate
                for candidate in self._live_running(requests)
                if candidate.prefix_group_id == request.prefix_group_id
            ]
            if len(group) == 1:
                self._prefill_one(request, result)
                handled.add(request.request_id)
                continue
            self._prefill_shared_group(group, result)
            handled.update(candidate.request_id for candidate in group)

    def _apply_prefix_cache_hit(self, request: Request) -> None:
        """Seed a fresh request's block table from the longest cached block-aligned prefix.

        Runs before prefill for a plain (non-group, un-started) request when the cross-turn
        prefix cache is on. A hit builds the request's table over the reused blocks and advances
        ``prompt_cached_tokens`` to the hit length, so ``_cache_prompt_chunk`` then prefills only
        the uncached tail via ``prefill_chunk(start_pos=k, ...)``. Because hits are block-aligned,
        the reused blocks stay read-only and every fresh write lands in a private block.
        """
        if self.prefix_cache is None:
            return
        if request.block_table is not None or request.prefix_group_id is not None:
            return
        hit = self.prefix_cache.lookup(request.prompt_ids)
        if hit is None:
            return
        table = self.cache.new_request()
        table.owner = request.request_id
        table.blocks = list(hit.block_ids)
        table.length = hit.tokens
        request.begin_prefill(table, cached_tokens=hit.tokens)

    def _can_prefill_many(self, requests: list[Request]) -> bool:
        return (
            self.batched_prefill
            and self.prefix_cache is None
            and len(requests) >= 2
            and self.capabilities.batched_prefill
            and isinstance(self.model, BatchedPrefillBackend)
            and self.prefill_chunk_size is None
            and not self.preemption
            and all(request.prefix_group_id is None for request in requests)
        )

    def _prefill_many(self, requests: list[Request], result: StepResult) -> None:
        assert isinstance(self.model, BatchedPrefillBackend)
        tables: list[BlockTable] = []
        for request in requests:
            state = self._prefilling(request)
            if state.table.length != 0 or state.cached_tokens != 0:
                raise ValueError(
                    f"request {request.request_id!r} must have an empty cache for batched prefill"
                )
            end_pos = len(request.prompt_ids)
            result.prefill_chunks[request.request_id] = (0, end_pos)
            self._trace_prefill_chunk_started(
                request_id=request.request_id,
                start_pos=0,
                end_pos=end_pos,
                total_prompt_tokens=end_pos,
            )
            tables.append(state.table)

        with self._record_time("prefill"):
            logits = self.model.prefill_many(
                [request.prompt_ids for request in requests], self.cache, tables
            )

        for request in requests:
            end_pos = len(request.prompt_ids)
            request.advance_prefill(end_pos)
            self._trace_prefill_chunk_progress(
                request_id=request.request_id,
                start_pos=0,
                end_pos=end_pos,
                cached_tokens=end_pos,
                total_prompt_tokens=end_pos,
                completed=True,
            )

        with self._record_time("sampling"):
            tokens = self._sample_rows(logits, requests)
        token_rows = list(tokens.unbind())
        eos_flags, host_ids = self._eos_flags_and_ids(tokens, requests)
        for request, token, is_eos, host_id in zip(
            requests, token_rows, eos_flags, host_ids, strict=True
        ):
            self._record(request, token, is_eos, result, host_token=host_id)
        self._trace_decode_step(requests, token_rows, token_source="prefill")
        self._release_finished_in(requests, result)

    def _prefill_one(self, request: Request, result: StepResult) -> None:
        logits = self._cache_prompt_chunk(request, result)
        if logits is None:
            return
        with self._record_time("sampling"):
            token = self._sample_one(logits, request)
        eos_flags, host_ids = self._eos_flags_and_ids(token.reshape(1), [request])
        self._record(request, token, eos_flags[0], result, host_token=host_ids[0])
        self._trace_decode_step([request], [token], token_source="prefill")
        self._release_finished_in([request], result)

    def _prefill_shared_group(self, requests: list[Request], result: StepResult) -> None:
        prompt_ids = requests[0].prompt_ids
        if any(request.prompt_ids != prompt_ids for request in requests):
            raise ValueError(
                f"prefix group {requests[0].prefix_group_id!r} contains different prompts"
            )

        leaders = [request for request in requests if request.block_table is not None]
        if len(leaders) > 1:
            raise ValueError(
                f"prefix group {requests[0].prefix_group_id!r} has multiple active leaders"
            )
        leader = leaders[0] if leaders else requests[0]
        logits = self._cache_prompt_chunk(leader, result)
        if logits is None:
            return

        # Leader prefill can preempt a sibling to make pool room (under preemption): that sibling
        # has been reset and returned to the waiting queue. Re-filter the group to the still-running
        # members before forking/sampling — forking onto a preempted sibling would resurrect it
        # out of the waiting queue with a live block table and a sampled token. It re-prefills
        # next step instead. The leader is never its own victim, so it always survives.
        members = [leader, *(r for r in self._live_running(requests) if r is not leader)]

        leader_state = leader.state
        if not isinstance(leader_state, Prefilling):
            raise ValueError("shared prefix leader must be prefilling")
        for request in members:
            if request is leader:
                continue
            table = self.cache.fork_request(leader_state.table)
            table.owner = request.request_id
            request.begin_prefill(table, cached_tokens=leader_state.cached_tokens)

        with self._record_time("sampling"):
            tokens = [self._sample_one(logits, request) for request in members]
        eos_flags, host_ids = self._eos_flags_and_ids(torch.stack(tokens), members)
        for request, token, is_eos, host_id in zip(
            members, tokens, eos_flags, host_ids, strict=True
        ):
            self._record(request, token, is_eos, result, host_token=host_id)
        self._trace_decode_step(members, tokens, token_source="prefill")
        self._release_finished_in(members, result)

    def _cache_prompt_chunk(self, request: Request, result: StepResult) -> torch.Tensor | None:
        """Cache one prompt chunk and return final-prompt logits when ready to sample."""
        state = self._prefilling(request)
        start_pos = state.cached_tokens
        if state.table.length != start_pos:
            raise ValueError(
                f"request {request.request_id!r} block table length "
                f"{state.table.length} != cached prompt length {start_pos}"
            )
        remaining = len(request.prompt_ids) - start_pos
        if remaining < 1:
            raise ValueError(f"request {request.request_id!r} has no prompt tokens left")

        chunk_size = self.prefill_chunk_size or len(request.prompt_ids)
        chunk_size = min(chunk_size, remaining)
        end_pos = start_pos + chunk_size
        if self.preemption:
            # A fresh prompt's prefill can also exhaust the pool — preempt LIFO victims so the
            # chunk's blocks are available before the model touches the cache.
            self._ensure_pool_room(request, new_tokens=chunk_size)
        result.prefill_chunks[request.request_id] = (start_pos, end_pos)
        self._trace_prefill_chunk_started(
            request_id=request.request_id,
            start_pos=start_pos,
            end_pos=end_pos,
            total_prompt_tokens=len(request.prompt_ids),
        )

        with self._record_time("prefill"):
            if start_pos == 0 and end_pos == len(request.prompt_ids):
                logits = self.model.prefill(request.prompt_ids, self.cache, state.table)
            else:
                logits = self.model.prefill_chunk(
                    request.prompt_ids,
                    self.cache,
                    state.table,
                    start_pos=start_pos,
                    chunk_size=chunk_size,
                )
        request.advance_prefill(end_pos)
        self._trace_prefill_chunk_progress(
            request_id=request.request_id,
            start_pos=start_pos,
            end_pos=end_pos,
            cached_tokens=end_pos,
            total_prompt_tokens=len(request.prompt_ids),
            completed=end_pos == len(request.prompt_ids),
        )
        if end_pos < len(request.prompt_ids):
            return None
        return logits

    def _prefilling(self, request: Request) -> Prefilling:
        state = request.state
        if isinstance(state, Pending):
            table = self.cache.new_request()
            table.owner = request.request_id
            return request.begin_prefill(table)
        if not isinstance(state, Prefilling):
            raise ValueError(f"request {request.request_id!r} is not awaiting prefill")
        return state

    def _decode_requests(self, requests: list[Request], result: StepResult) -> None:
        """Advance decode-ready requests, optionally using prompt-lookup speculation."""
        if self.preemption:
            requests = self._make_decode_room(requests)
            if not requests:
                return
        if self._window_eligible(requests):
            self._decode_window_step(requests, result)
            return
        # Leaving the deferred path (e.g. a penalty-carrying request joined the batch): drain
        # the open window first so every request's recorded state is current, then decode the
        # survivors.
        self._drain_decode_window(result)
        requests = [request for request in requests if not request.finished]
        if not requests:
            return
        if self.speculative is None:
            self._decode_normal(requests, result)
            return

        fallback: list[Request] = []
        for request in requests:
            draft = self._draft_for(request)
            if not draft:
                fallback.append(request)
                continue
            self._decode_speculative(request, draft, result)

        if fallback:
            self._decode_normal(fallback, result)

    def _tables_for(self, requests: list[Request]) -> list[BlockTable]:
        tables: list[BlockTable] = []
        for request in requests:
            state = request.state
            if not isinstance(state, DecodeReady):
                raise ValueError(f"request {request.request_id!r} is not decode-ready")
            tables.append(state.table)
        return tables

    def _decode_normal(self, requests: list[Request], result: StepResult) -> None:
        """The original one-token batched decode path: shared forward, per-row sampling."""
        last_tokens = torch.stack([request.last_token_tensor for request in requests]).to(
            self.model.device
        )
        with self._record_time("decode"):
            logits = self.model.decode_many(
                self.cache,
                self._tables_for(requests),
                last_tokens,
            )
        with self._record_time("sampling"):
            batched = self._sample_rows(logits, requests)
        tokens = list(batched.unbind(0))
        eos_flags, host_ids = self._eos_flags_and_ids(batched, requests)
        for request, token, is_eos, host_id in zip(
            requests, tokens, eos_flags, host_ids, strict=True
        ):
            self._record(request, token, is_eos, result, host_token=host_id)
        self._trace_decode_step(requests, list(tokens), token_source="decode")
        self._release_finished_in(requests, result)

    def _window_eligible(self, requests: list[Request]) -> bool:
        """Whether this decode batch may run in a deferred window (device-side stop tracking).

        The window trades per-step host syncs for one sync per ``budget`` steps, which is only
        safe when nothing per step needs host data: every row greedy or penalty-free sampled
        (a penalty reads the row's generated history each step; a penalty-free draw needs only
        its device-resident generator), no speculation (drafts read generated ids), no
        preemption (victim selection inspects live state), and no tracing (events are
        per-token). Prefix-group rows may re-enter once their block table is private, so a
        finisher cannot leave shared blocks referenced past its recorded EOS.
        """
        return (
            self.decode_window_size > 1
            and self.speculative is None
            and not self.preemption
            and self.trace is None
            and self._window_blocks_are_private(requests)
            and all(self._window_row_eligible(request) for request in requests)
        )

    def _window_row_eligible(self, request: Request) -> bool:
        """Greedy rows and penalty-free sampled rows can decode without per-step host data."""
        params = self._params_for(request)
        return params.is_greedy or (
            params.presence_penalty == 0.0 and params.frequency_penalty == 0.0
        )

    def _window_blocks_are_private(self, requests: list[Request]) -> bool:
        """The planned window can skip copy-on-write only when every live block is private.

        ``build_decode_window_plan`` refuses *any* shared block in a table, not just shared write
        targets, and the deferred window keeps a mid-window finisher's blocks referenced past its
        recorded EOS. A cross-turn cache-hit request keeps its reused prompt-prefix blocks shared
        with the store (``prefix_group_id`` is ``None`` but the blocks are not private), so when the
        cache is on every request's blocks are inspected; such a request falls back to the per-step
        decode path. With neither prefix source active the blocks are always private and the
        refcount walk is skipped for the hot path.
        """
        check_all = self.prefix_cache is not None
        for request in requests:
            state = request.state
            if not isinstance(state, DecodeReady):
                return False
            table = state.table
            if request.prefix_group_id is None and not check_all:
                continue
            if any(self.cache.allocator.refcount(block) > 1 for block in table.blocks):
                return False
        return True

    def _decode_window_step(self, requests: list[Request], result: StepResult) -> None:
        """One deferred decode step: batched forward + on-device sampling, no host sync.

        The previous step's token tensor feeds the next forward directly, so within a window
        the loop never materializes tokens on the host. The window flushes when its budget is
        reached — a *staged* non-blocking copy, consumed one window later, so consecutive
        same-batch windows never block on the compute stream. A batch-composition change
        (admission or a consume-released finisher) drains staged state first and re-filters
        the batch.
        """
        window = self._decode_windows.active
        ids = tuple(request.request_id for request in requests)
        if window is not None and window.request_ids != ids:
            self._drain_decode_window(result)
            window = None
            requests = [request for request in requests if not request.finished]
            if not requests:
                return
            ids = tuple(request.request_id for request in requests)
        if window is None:
            pending = self._decode_windows.pending
            if pending is not None:
                # The staged window's tokens are not recorded yet, so the next window may
                # pipeline behind it only when the batch is identical and no request could
                # pass its length cap counting those staged steps. Otherwise consume now.
                pipelined = pending.request_ids == ids and all(
                    request.remaining_tokens - pending.steps >= 1 for request in requests
                )
                if not pipelined:
                    self._drain_decode_window(result)
                    pending = None
                    requests = [request for request in requests if not request.finished]
                    if not requests:
                        return
                    ids = tuple(request.request_id for request in requests)
            staged_steps = pending.steps if pending is not None else 0
            window = DecodeWindow(
                request_ids=ids,
                requests=list(requests),
                budget=min(
                    self.decode_window_size,
                    min(request.remaining_tokens for request in requests) - staged_steps,
                ),
                all_greedy=all(self._params_for(request).is_greedy for request in requests),
            )
            if self.capabilities.planned_decode:
                assert self._planned_backend is not None
                window.plan = self._planned_backend.open_decode_window(
                    self.cache,
                    self._tables_for(requests),
                    window.budget,
                )
            self._decode_windows.active = window

        if window.pending:
            last_tokens = window.pending[-1]
        else:
            pending = self._decode_windows.pending
            if pending is not None and pending.request_ids == window.request_ids:
                # Pipelined windows: the previous window's last sampled tokens are still on
                # device (and not yet recorded) — feed them straight into this window.
                last_tokens = pending.matrix[-1]
            else:
                last_tokens = torch.stack([request.last_token_tensor for request in requests]).to(
                    self.model.device
                )
        with self._record_time("decode"):
            if window.plan is not None:
                # Dispatch chain: grouped runner (exact batch, this cache) → the model's
                # piecewise runner (padded bucket) → the eager planned step. The grouped
                # runner declines before touching the plan, so exactly one link advances it.
                logits = self._grouped_window_step(window.plan, last_tokens)
                if logits is None:
                    assert self._planned_backend is not None
                    logits = self._planned_backend.decode_window_step(
                        self.cache, window.plan, last_tokens
                    )
            else:
                logits = self.model.decode_many(
                    self.cache,
                    self._tables_for(requests),
                    last_tokens,
                )
        with self._record_time("sampling"):
            if window.all_greedy:
                window.pending.append(torch.argmax(logits, dim=-1))
            else:
                window.pending.append(self._sample_rows(logits, window.requests))
        if len(window.pending) >= window.budget:
            self._flush_decode_window(result)

    def _grouped_window_step(
        self, plan: DecodeWindowPlan, token_ids: torch.Tensor
    ) -> torch.Tensor | None:
        """Try the engine-owned grouped runner for this exact batch size, or ``None``.

        A runner exists only for its captured batch size and only against the engine's own
        cache (it re-checks the cache identity and KV pointer itself), so a miss here is
        the expected path, not an error — the caller falls through to the piecewise bucket.
        """
        runner = self.grouped_decode_runners.get(len(plan.tables))
        if runner is None:
            return None
        return runner.window_step(self.cache, plan, token_ids)

    def _flush_decode_window(self, result: StepResult) -> None:
        """Close the open window: stage its host copy, then consume the *previous* stage.

        The stage is one ``non_blocking`` device-to-host copy of (tokens, EOS flags) into
        pinned buffers plus an event — no wait on the compute stream. The previously staged
        window, whose copy has been in flight for a whole window of GPU work, is consumed
        into request state here. Tokens after a request's first EOS (or its length cap) are
        decode overshoot and are discarded at consume; the tokens kept are exactly what the
        per-step path would have recorded, so outputs stay token-for-token identical to the
        classic loop.
        """
        previous = self._decode_windows.flush()
        if previous is not None:
            if self.profiler is None:
                self._consume_window_flush(previous, result)
            else:
                with self.profiler.host("window_flushing"):
                    self._consume_window_flush(previous, result)

    def _drain_decode_window(self, result: StepResult) -> None:
        """Consume every staged copy before changing batch composition or aborting."""
        for pending in self._decode_windows.drain():
            if self.profiler is None:
                self._consume_window_flush(pending, result)
            else:
                with self.profiler.host("window_flushing"):
                    self._consume_window_flush(pending, result)

    def _consume_window_flush(self, pending: PendingWindowFlush, result: StepResult) -> None:
        """Record a staged window into request state and release its finishers."""
        if pending.event is not None:
            with self._record_host_time("cpu_gpu_sync"):
                pending.event.synchronize()
        step_ids = pending.tokens_host.tolist()  # host ints for EOS/stop decisions
        eos_rows = pending.eos_host.tolist() if pending.eos_host is not None else None
        for column, request in enumerate(pending.requests):
            for step in range(pending.steps):
                if request.finished:
                    break
                is_eos = (
                    eos_rows[step][column]
                    if eos_rows is not None
                    else step_ids[step][column] in request.eos_token_ids
                )
                # Record the device-tensor view (keeps a request's generated tokens on one
                # device — ``Request.generated`` stacks them); the host flag drives the stop
                # rule, and the already-copied host id rides along so the serving dispatcher
                # never re-syncs per token.
                self._record(
                    request,
                    pending.matrix[step, column],
                    is_eos,
                    result,
                    host_token=step_ids[step][column],
                )
        self._release_finished_in(pending.requests, result)

    def _params_for(self, request: Request) -> SamplingParams:
        """The request's own sampling params, or the engine default when it set none."""
        return request.sampling if request.sampling is not GREEDY else self.default_sampling

    def _sample_one(self, logits: torch.Tensor, request: Request) -> torch.Tensor:
        """Sample one token from a 1-D ``(vocab,)`` row under this request (prefill path)."""
        return sample_row(
            logits, self._params_for(request), request.generated, request.generator(logits.device)
        )

    def _sample_rows(self, logits: torch.Tensor, requests: list[Request]) -> torch.Tensor:
        """Sample one token per row of ``(B, vocab)`` logits, each under its own request.

        An all-greedy batch — the benchmark and reference path — is one batched argmax, no
        per-row Python at all. Otherwise :func:`sample_rows` runs the filtering surface batched
        on the logits' device and draws one token per sampled row from that request's own seeded
        generator, so batchmates cannot consume its RNG stream. Generated history is
        materialized (a host sync) only for rows that actually carry a penalty.
        """
        params = [self._params_for(request) for request in requests]
        generators = [
            None if p.is_greedy else request.generator(logits.device)
            for request, p in zip(requests, params, strict=True)
        ]
        histories = [
            request.generated if p.presence_penalty != 0.0 or p.frequency_penalty != 0.0 else None
            for request, p in zip(requests, params, strict=True)
        ]
        return sample_rows(logits, params, generators, histories)

    def _decode_budget(self, request: Request) -> int:
        """Worst-case tokens this request may append in one decode step — for room reservation.

        A speculative-eligible (greedy) request verifies ``last_token`` plus up to
        ``max_draft_tokens`` in one forward and reserves blocks for all of them, so room must
        cover the whole draft or the verify can OOM mid-step; the bound mirrors ``_draft_for``
        (capped by the request's remaining tokens). Every other request appends exactly one token.
        """
        if self.speculative is not None and self._params_for(request).is_greedy:
            max_draft = min(
                self.speculative.config.max_draft_tokens, max(0, request.remaining_tokens - 1)
            )
            return 1 + max_draft
        return 1

    def _make_decode_room(self, requests: list[Request]) -> list[Request]:
        """Ensure the whole decode batch can grow by its per-request budget; preempt LIFO if not.

        ``decode_many`` allocates for every surviving member in one call, so room must cover the
        batch's *total* growth, not one request at a time. A normal request appends one token (at
        most one new block); a speculative one may append its whole draft, so each is reserved for
        ``_decode_budget`` tokens. A victim is the most-recently-admitted running request and may
        itself be in this batch — preempting it drops it from the step and lowers the demand. We
        preempt until the free pool covers the survivors' combined growth.

        Forward progress holds: the block-needers are the batch members, and preempting strictly
        shrinks the batch, so the demand reaches zero before victims run out.
        """
        survivors = [r for r in requests if isinstance(r.state, DecodeReady)]
        while True:
            survivors = [r for r in survivors if isinstance(r.state, DecodeReady)]
            demand = sum(self._blocks_to_grow(r, self._decode_budget(r)) for r in survivors)
            if demand <= self.cache.allocator.num_free:
                return survivors
            victim = self.scheduler.preemption_victim(exclude=None)
            if victim is None:
                raise OutOfBlocksError(
                    "decode batch needs more blocks than the pool can free — the "
                    "forward-progress invariant was violated"
                )
            self._preempt(victim)

    def _draft_for(self, request: Request) -> list[int]:
        """Return a draft only when there is room for draft tokens plus verifier recovery.

        Speculative decoding verifies with a greedy (argmax) verifier, so only a greedy request
        is eligible: a sampled request falls through to the normal per-row sampling path, which
        keeps its draw seeded and independent. The guard is per request, not engine-wide, so a
        greedy request can still speculate while a sampled one in the same batch does not.
        """
        if self.speculative is None or not self._params_for(request).is_greedy:
            return []
        max_draft_tokens = request.remaining_tokens - 1
        if max_draft_tokens < 1:
            return []
        return self.speculative.draft(
            request.prompt_ids + request.generated,
            max_tokens=max_draft_tokens,
        )

    def _decode_speculative(self, request: Request, draft: list[int], result: StepResult) -> None:
        """Verify one request's draft and emit the accepted prefix plus recovery token."""
        state = request.state
        if not isinstance(state, DecodeReady):
            raise ValueError(f"request {request.request_id!r} is not decode-ready")
        table = state.table
        original_length = table.length
        draft_tensor = torch.tensor(draft, dtype=torch.long, device=self.model.device)
        verify_input = torch.cat(
            [
                request.last_token_tensor.to(self.model.device).reshape(1),
                draft_tensor,
            ]
        )
        with self._record_time("speculative_decode"):
            logits = self.model.decode_tokens(self.cache, table, verify_input)

        verifier_tokens = torch.argmax(logits, dim=-1)
        accepted = self._accepted_prefix_length(verifier_tokens[:-1], draft_tensor)

        # Record accepted draft tokens as views of the on-device draft tensor, not the Python
        # ints they were drafted from: a request's generated tokens must all live on the model
        # device (``Request.generated`` stacks them; a CPU/CUDA mix would raise there).
        emitted: list[int | torch.Tensor] = list(draft_tensor[:accepted].unbind())
        if not self._contains_eos(emitted, request):
            if accepted == len(draft):
                emitted.append(verifier_tokens[-1])
            else:
                emitted.append(verifier_tokens[accepted])

        emitted = self._truncate_after_eos(emitted, request)
        if not emitted:
            raise ValueError("speculative verification produced no token to emit")

        table.length = min(original_length + len(emitted), table.length)
        for token in emitted:
            if request.finished:
                break
            self._record(request, token, self._is_eos(token, request), result)
        self._trace_decode_step([request], emitted, token_source="speculative")
        # Verification reserved blocks for the whole draft; a partly-rejected draft rolled the
        # length back, so return the unused trailing blocks to the pool (a finisher's table is
        # freed wholesale by _release_finished_in, so only trim a still-running request).
        if not request.finished:
            table.trim_to_length()
        self._release_finished_in([request], result)

    def _accepted_prefix_length(
        self, verifier_tokens: torch.Tensor, draft_tokens: torch.Tensor
    ) -> int:
        """Length of the contiguous draft prefix matched by greedy verifier tokens."""
        matches = verifier_tokens == draft_tokens
        with self._record_host_time("cpu_gpu_sync"):
            flags = [bool(flag) for flag in matches.cpu().tolist()]
        accepted = 0
        for flag in flags:
            if not flag:
                break
            accepted += 1
        return accepted

    def _record(
        self,
        request: Request,
        token: int | torch.Tensor,
        is_eos: bool,
        result: StepResult,
        host_token: int | None = None,
    ) -> None:
        """Append a sampled token to a request and note it (and any finish) in the step result.

        ``token`` may be a device tensor — request state keeps it on the model device. When the
        caller already holds the token's host id (every path that synced for EOS anyway),
        passing it as ``host_token`` puts a plain int in the step result, so the serving
        dispatcher never pays a per-token device sync to materialize it again.
        """
        request.record(token, is_eos=is_eos)
        result.tokens.setdefault(request.request_id, []).append(
            host_token if host_token is not None else token
        )
        if request.finished and request.request_id not in result.finished:
            result.finished.append(request.request_id)

    def _release_finished_in(self, requests: list[Request], result: StepResult) -> None:
        """Free and release any of ``requests`` that just finished, promptly.

        Called at the end of each prefill/decode op — after its ``decode_step`` trace, so the
        finish events stay ordered after the token that produced them. Releasing here rather than
        at a single end-of-step sweep returns a finisher's KV before the *next* prefill/decode
        this step needs room, and takes it out of the running set so it can never be selected as a
        preemption victim (recompute refuses a finished request). Resume samples nothing and never
        finishes, so prefill and decode together cover every finish source.
        """
        for request in requests:
            if request.finished and self._is_running(request):
                self._trace_request_finished(request)
                result.finished_outputs[request.request_id] = request.generated
                if request.block_table is not None:
                    # Donate the block-aligned prompt-plus-output prefix to the cross-turn cache
                    # before freeing: the cache retains those blocks, so the table free only drops
                    # this request's ref and the donated blocks stay alive for a later turn.
                    if self.prefix_cache is not None:
                        self._donate_to_prefix_cache(request)
                    self._free_block_table(request.block_table)
                request.release_finished_table()
                self.scheduler.release(request)
                self._requests.pop(request.request_id, None)

    def _donate_to_prefix_cache(self, request: Request) -> None:
        """Cache a finished request's block-aligned cached prefix for cross-turn reuse.

        Deferred windows can cache tokens beyond recorded EOS. Donate complete blocks only
        within the recorded sequence, excluding the final sampled token's uncached position.
        """
        if self.prefix_cache is None:
            return
        table = request.block_table
        if table is None:
            return
        block_size = self.cache.block_size
        recorded_positions = len(request.prompt_ids) + request.generated_count - 1
        cached_tokens = (min(table.length, recorded_positions) // block_size) * block_size
        if cached_tokens < block_size:
            return
        token_ids = (request.prompt_ids + request.generated)[:cached_tokens]
        block_ids = table.blocks[: cached_tokens // block_size]
        self.prefix_cache.donate(token_ids, block_ids)

    def _free_block_table(self, table: BlockTable) -> None:
        """Release backend per-table state, then return physical blocks to the pool."""
        self.model.release_table(table)
        table.free()

    def _eos_flags_and_ids(
        self, tokens: torch.Tensor, requests: list[Request]
    ) -> tuple[list[bool], list[int]]:
        """Per-request EOS flags plus host token ids, from one device-to-host copy.

        The per-step paths must sync for the stop rule anyway; copying the token ids
        themselves (instead of a device-side EOS mask) makes that single sync also yield the
        plain ints the serving dispatcher needs, and the tiny stop-set membership check is
        cheaper on the host than as extra device kernels.
        """
        flat = tokens.reshape(-1)
        if len(flat) != len(requests):
            raise ValueError(f"token/request count mismatch: {len(flat)} vs {len(requests)}")
        with self._record_host_time("cpu_gpu_sync"):
            ids = [int(token_id) for token_id in flat.cpu().tolist()]
        flags = [
            token_id in request.eos_token_ids
            for token_id, request in zip(ids, requests, strict=True)
        ]
        return flags, ids

    def _is_eos(self, token: int | torch.Tensor, request: Request) -> bool:
        if isinstance(token, torch.Tensor):
            with self._record_host_time("cpu_gpu_sync"):
                token_id = int(token.cpu().item())
        else:
            token_id = token
        return token_id in request.eos_token_ids

    def _contains_eos(self, tokens: list[int | torch.Tensor], request: Request) -> bool:
        return any(self._is_eos(token, request) for token in tokens)

    def _truncate_after_eos(
        self, tokens: list[int | torch.Tensor], request: Request
    ) -> list[int | torch.Tensor]:
        truncated: list[int | torch.Tensor] = []
        for token in tokens:
            truncated.append(token)
            if self._is_eos(token, request):
                break
        return truncated

    def _blocks_to_grow(self, request: Request, new_tokens: int) -> int:
        """How many *new* physical blocks ``request`` must pull to cache ``new_tokens`` more.

        Delegates to the cache's dry-run cost so the count includes copy-on-write: a request
        whose last block is a prefix-shared partial block copies it private on append, which
        pulls a block the plain capacity-growth math misses (and would otherwise OOM mid-write).
        """
        table = request.block_table
        if table is None:
            # Fresh request: no blocks yet and nothing shared, so just capacity growth from empty.
            return -(-new_tokens // self.scheduler.block_size)  # ceil division
        return self.cache.append_cost(table, new_tokens)

    def _ensure_pool_room(self, request: Request, new_tokens: int) -> None:
        """Free enough blocks for ``request`` to grow by ``new_tokens``, preempting LIFO victims.

        Forward-progress invariant: a request fits in the empty pool alone (admission rejects
        any that does not), so once every *other* running request is preempted ``request`` can
        always grow — the loop cannot spin forever. ``request`` is the block-needer and is passed
        as ``exclude`` so it never evicts itself; we stop once the free pool covers the growth.
        """
        need = self._blocks_to_grow(request, new_tokens)
        while self.cache.allocator.num_free < need:
            victim = self.scheduler.preemption_victim(exclude=request)
            if victim is None:
                raise OutOfBlocksError(
                    f"request {request.request_id!r} needs {need} block(s) to grow but the "
                    f"pool has {self.cache.allocator.num_free} free and no other request to "
                    "preempt — the forward-progress invariant was violated"
                )
            self._preempt(victim)

    def _preempt(self, victim: Request) -> None:
        """Evict ``victim`` by recompute: free its KV clearly, keep its tokens, requeue it."""
        self.preemption_count += 1
        freed_blocks = len(victim.block_table.blocks) if victim.block_table is not None else 0
        if victim.block_table is not None:
            self._free_block_table(victim.block_table)
        victim.reset_for_recompute()
        self.scheduler.requeue(victim)
        self._emit_trace(
            "request_preempted",
            request_id=victim.request_id,
            preempt_reason="kv_pressure",
            block_count=freed_blocks,
            generated_tokens=victim.generated_count,
            pool_used=self.cache.allocator.num_used,
            pool_free=self.cache.allocator.num_free,
        )

    def _is_running(self, request: Request) -> bool:
        """Whether ``request`` is in the running set *right now*, tested by identity.

        Prefilling or resuming an earlier request can preempt a later candidate via
        ``_ensure_pool_room``, moving it to ``waiting`` partway through the loop. Both loops
        therefore re-check membership live rather than against a set snapshotted once before the
        loop: a stale snapshot would let an already-evicted request be prefilled while it sits in
        ``waiting``, double-allocating its KV and breaking forward progress.
        """
        return self.scheduler.is_running(request)

    def _live_running(self, requests: Iterable[Request]) -> list[Request]:
        """Filter a candidate list against live running membership under preemption.

        Without preemption, a step's request lists stay valid for the whole step. With
        preemption, an earlier operation can evict a later candidate, so loops that act
        after allocation pressure re-check membership at the point of use.
        """
        if not self.preemption:
            return list(requests)
        return [request for request in requests if self._is_running(request)]

    def _resume_requests(self, requests: list[Request], result: StepResult) -> None:
        """Rebuild each preempted request's KV by recompute before it decodes again.

        Recompute, not swap: the request kept its generated tokens through preemption, so a
        fresh prefill over ``recompute_prompt_ids`` reconstructs exactly the KV state it was
        evicted in. No token is sampled — the request already owns its generated ids — so on the
        next step it decodes its next token, identical to the uninterrupted run.
        """
        for request in self._live_running(requests):
            self._recompute_prefill(request)
            self._emit_trace(
                "request_resumed",
                request_id=request.request_id,
                prompt_tokens=len(request.prompt_ids),
                generated_tokens=request.generated_count,
                cached_tokens=request.prompt_cached_tokens,
                pool_used=self.cache.allocator.num_used,
                pool_free=self.cache.allocator.num_free,
            )

    def _recompute_prefill(self, request: Request) -> None:
        """Re-prefill the request's pre-feed sequence into a fresh table, sampling nothing.

        The sequence is ``prompt + generated[:-1]`` (see ``Request.recompute_prompt_ids``): the
        last generated token is *not* written here, it is re-fed by the resuming decode at its
        original position. Walks the sequence in the same bounded chunks as a normal prefill (so
        a long resume shares the loop and can itself be re-preempted between chunks), rebuilding
        KV at the original absolute positions. The final logits are discarded — the request's
        generated tokens already determine what it decodes next.
        """
        sequence = request.recompute_prompt_ids
        state = self._prefilling(request)
        if state.mode != "recompute":
            raise ValueError(f"request {request.request_id!r} does not need recompute")

        total = len(sequence)
        chunk_size = self.prefill_chunk_size or total
        while state.cached_tokens < total:
            start = state.cached_tokens
            step = min(chunk_size, total - start)
            self._ensure_pool_room(request, step)
            self._trace_prefill_chunk_started(
                request_id=request.request_id,
                start_pos=start,
                end_pos=start + step,
                total_prompt_tokens=total,
            )
            with self._record_time("prefill"):
                self._write_recompute_chunk(state.table, sequence, start, step)
            request.advance_prefill(start + step)
            state = self._prefilling(request)
            self._trace_prefill_chunk_progress(
                request_id=request.request_id,
                start_pos=start,
                end_pos=start + step,
                cached_tokens=request.prompt_cached_tokens,
                total_prompt_tokens=total,
                completed=request.prompt_cached_tokens == total,
            )
        request.finish_recompute()

    def _write_recompute_chunk(
        self, table: BlockTable, sequence: list[int], start: int, count: int
    ) -> None:
        """Write KV for ``sequence[start:start+count]`` via the model's cached prefill path."""
        if start == 0 and count == len(sequence):
            self.model.prefill(sequence, self.cache, table)
            return
        self.model.prefill_chunk(
            sequence,
            self.cache,
            table,
            start_pos=start,
            chunk_size=count,
        )

    def _trace_decode_step(
        self,
        requests: list[Request],
        tokens: Sequence[int | torch.Tensor],
        *,
        token_source: TokenSource,
    ) -> None:
        if self.trace is None:
            return
        self._emit_trace(
            "decode_step",
            request_ids=tuple(request.request_id for request in requests),
            batch_size=len(requests),
            token_ids=tuple(self._trace_token_id(token) for token in tokens),
            tokens_emitted=len(tokens),
            token_source=token_source,
        )

    def _trace_prefill_chunk_started(
        self, request_id: str, *, start_pos: int, end_pos: int, total_prompt_tokens: int
    ) -> None:
        self._emit_trace(
            "prefill_chunk_started",
            request_id=request_id,
            start_pos=start_pos,
            end_pos=end_pos,
            total_prompt_tokens=total_prompt_tokens,
        )

    def _trace_prefill_chunk_progress(
        self,
        request_id: str,
        *,
        start_pos: int,
        end_pos: int,
        cached_tokens: int,
        total_prompt_tokens: int,
        completed: bool,
    ) -> None:
        self._emit_trace(
            "prefill_chunk_progress",
            request_id=request_id,
            start_pos=start_pos,
            end_pos=end_pos,
            cached_tokens=cached_tokens,
            total_prompt_tokens=total_prompt_tokens,
            completed=completed,
        )

    def _trace_pool_event(self, event: BlockPoolEvent) -> None:
        """Emit block lifecycle from the allocator's physical free-pool boundary.

        Each :class:`BlockPoolEvent` is purely an allocation or a real free (refcount-0):
        ``allocate`` reports no frees, ``free`` reports only the blocks that returned to the
        pool, and ``retain`` (prefix sharing) reports nothing at all.
        """
        if event.allocated:
            self._emit_trace(
                "block_allocated",
                request_id=event.owner,
                block_count=len(event.allocated),
                block_ids=event.allocated,
                pool_used=event.num_used,
                pool_free=event.num_free,
            )
        if event.freed:
            self._emit_trace(
                "block_freed",
                request_id=event.owner,
                block_count=len(event.freed),
                block_ids=event.freed,
                pool_used=event.num_used,
                pool_free=event.num_free,
            )

    def _trace_request_finished(self, request: Request) -> None:
        if self.trace is None:
            return
        reason: FinishReason = "eos" if request.last_token in request.eos_token_ids else "length"
        self._emit_trace(
            "request_finished",
            request_id=request.request_id,
            token_ids=tuple(request.generated),
            generated_tokens=request.generated_count,
            reason=reason,
        )

    def _trace_batch_size_changed(self) -> None:
        if self.trace is None:
            return
        batch_size = len(self.scheduler.running)
        if batch_size == self._last_traced_batch_size:
            return
        self._emit_trace(
            "batch_size_changed",
            previous_batch_size=self._last_traced_batch_size,
            batch_size=batch_size,
            waiting=len(self.scheduler.waiting),
        )
        self._last_traced_batch_size = batch_size

    def _trace_throughput_sample(self, result: StepResult) -> None:
        if self.trace is None:
            return
        tokens_this_step = sum(len(tokens) for tokens in result.tokens.values())
        if tokens_this_step == 0:
            return
        self._trace_total_tokens += tokens_this_step
        elapsed = max(self._trace_clock() - self._trace_start_time, 1e-12)
        self._emit_trace(
            "tokens_per_second_sampled",
            tokens_emitted=tokens_this_step,
            total_generated_tokens=self._trace_total_tokens,
            elapsed_seconds=elapsed,
            tokens_per_second=self._trace_total_tokens / elapsed,
        )

    def _emit_trace(
        self,
        event: TraceEventName,
        *,
        request_id: str | None = None,
        request_ids: tuple[str, ...] = (),
        prompt_tokens: int | None = None,
        max_new_tokens: int | None = None,
        prefix_group_id: str | None = None,
        reserved_blocks: int | None = None,
        start_pos: int | None = None,
        end_pos: int | None = None,
        cached_tokens: int | None = None,
        total_prompt_tokens: int | None = None,
        completed: bool | None = None,
        block_count: int | None = None,
        block_ids: tuple[int, ...] = (),
        pool_used: int | None = None,
        pool_free: int | None = None,
        batch_size: int | None = None,
        previous_batch_size: int | None = None,
        waiting: int | None = None,
        token_ids: tuple[int, ...] = (),
        tokens_emitted: int | None = None,
        token_source: TokenSource | None = None,
        generated_tokens: int | None = None,
        reason: FinishReason | None = None,
        preempt_reason: PreemptReason | None = None,
        total_generated_tokens: int | None = None,
        elapsed_seconds: float | None = None,
        tokens_per_second: float | None = None,
    ) -> None:
        if self.trace is None:
            return
        if self._trace_step is None:
            raise RuntimeError("trace events can only be emitted during engine.step()")
        self._trace_sequence += 1
        self.trace.record(
            TraceEvent(
                event=event,
                sequence=self._trace_sequence,
                step=self._trace_step,
                request_id=request_id,
                request_ids=request_ids,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                prefix_group_id=prefix_group_id,
                reserved_blocks=reserved_blocks,
                start_pos=start_pos,
                end_pos=end_pos,
                cached_tokens=cached_tokens,
                total_prompt_tokens=total_prompt_tokens,
                completed=completed,
                block_count=block_count,
                block_ids=block_ids,
                pool_used=pool_used,
                pool_free=pool_free,
                batch_size=batch_size,
                previous_batch_size=previous_batch_size,
                waiting=waiting,
                token_ids=token_ids,
                tokens_emitted=tokens_emitted,
                token_source=token_source,
                generated_tokens=generated_tokens,
                reason=reason,
                preempt_reason=preempt_reason,
                total_generated_tokens=total_generated_tokens,
                elapsed_seconds=elapsed_seconds,
                tokens_per_second=tokens_per_second,
            )
        )

    def _trace_token_id(self, token: int | torch.Tensor) -> int:
        if isinstance(token, torch.Tensor):
            with self._record_host_time("cpu_gpu_sync"):
                return int(token.cpu().item())
        return token

    def _record_time(self, name: str) -> AbstractContextManager[None]:
        if self.profiler is None:
            return nullcontext()
        return self.profiler.record(name)

    def _record_host_time(self, name: str) -> AbstractContextManager[None]:
        if self.profiler is None:
            return nullcontext()
        return self.profiler.host(name)
