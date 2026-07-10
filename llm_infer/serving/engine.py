"""Continuous-batching decode loop: one step advances decode and prefill work."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import torch

from llm_infer.kernels.base import PackedPrefillAttentionBackend
from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.grouped_decode_graph import (
    DEFAULT_GROUPED_CAPTURE_SIZES,
    EngineOwnedGroupedDecodeGraphRunner,
    build_engine_grouped_runners,
)
from llm_infer.model.interface import (
    DENSE_CAPABILITIES,
    QWEN_CAPABILITIES,
    BackendCapabilities,
    CausalLMBackend,
)
from llm_infer.model.pretrain_bundle import PretrainBundleModel
from llm_infer.profiling import TimingProfiler
from llm_infer.scheduler.scheduler import Scheduler
from llm_infer.serving.engine_decode import DecodeWindow, EngineDecodeMixin, PendingWindowFlush
from llm_infer.serving.engine_preemption import EnginePreemptionMixin
from llm_infer.serving.engine_prefill import EnginePrefillMixin
from llm_infer.serving.engine_trace import EngineTraceMixin
from llm_infer.serving.request import Request
from llm_infer.serving.sampler import GREEDY, SamplingParams
from llm_infer.serving.speculative import PromptLookupDraft, SpeculativeDecodingConfig
from llm_infer.tracing import TraceRecorder


@dataclass
class StepResult:
    """What happened in one engine step — enough to trace the vertical slice."""

    admitted: list[str] = field(default_factory=list)
    prefill_chunks: dict[str, tuple[int, int]] = field(default_factory=dict)
    finished: list[str] = field(default_factory=list)
    tokens: dict[str, list[int | torch.Tensor]] = field(default_factory=dict)
    finished_outputs: dict[str, list[int]] = field(default_factory=dict)


class InferenceEngine(
    EngineTraceMixin,
    EnginePreemptionMixin,
    EnginePrefillMixin,
    EngineDecodeMixin,
):
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
        self.capabilities = capabilities or _infer_capabilities(model)
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
        # Per-stop-set EOS comparison tensors, built once and reused across decode steps.
        self._eos_tensors: dict[frozenset[int], torch.Tensor] = {}
        # Deferred-decode window state: how many one-token decode steps may run between EOS/stop
        # host boundaries (1 = classic per-step sync), the currently open window, and the
        # staged flush of the previous window (consumed one window behind; see
        # PendingWindowFlush in engine_decode).
        self.decode_window_size = decode_window_size
        self._decode_window: DecodeWindow | None = None
        self._pending_flush: PendingWindowFlush | None = None
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
            if not self.capabilities.planned_decode or not isinstance(model, PretrainBundleModel):
                raise ValueError(
                    "grouped_decode_graphs requires a bundle model with planned decode; "
                    "omit it for this backend"
                )
            if grouped_layers is None:
                # Four grouped layers is the measured A100 winner; the runner needs a
                # following ungrouped layer, so very small (test) models group two.
                grouped_layers = 4 if model.num_layers > 4 else 2
            self.grouped_decode_runners = build_engine_grouped_runners(
                model,
                self.cache,
                grouped_capture_sizes,
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

    def add_request(self, request: Request) -> None:
        """Register and queue a request. Duplicate ids are rejected loudly."""
        if request.prefix_group_id is not None and not self.capabilities.prefix_caching:
            raise ValueError(
                "prefix_group_id requires prefix caching; this backend does not support it"
            )
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request id {request.request_id!r}")
        self._requests[request.request_id] = request
        self.scheduler.add(request)

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
        if self._decode_window is not None or self._pending_flush is not None:
            stashed = self._stashed_flush or StepResult()
            self._drain_decode_window(stashed)
            self._stashed_flush = stashed
        request = self._requests.pop(request_id, None)
        if request is None:
            return False
        if request in self.scheduler.waiting:
            self.scheduler.waiting.remove(request)
            return True
        if request in self.scheduler.running:
            if request.block_table is not None:
                self._free_block_table(request.block_table)
            self.scheduler.release(request)
            return True
        return False

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
                if request.prefilled:
                    to_decode.append(request)
                elif request.generated:
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
                while self._decode_window is not None and not self.scheduler.waiting:
                    self._decode_window_step(self._decode_window.requests, result)

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


def _infer_capabilities(model: CausalLMBackend) -> BackendCapabilities:
    if isinstance(model, PretrainBundleModel):
        return replace(
            DENSE_CAPABILITIES,
            batched_prefill=isinstance(model.backend, PackedPrefillAttentionBackend),
        )
    return QWEN_CAPABILITIES
