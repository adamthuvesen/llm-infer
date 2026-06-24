"""The minimal continuous-batching decode loop — the Phase B vertical slice runner.

Wires the model, the paged KV-cache, and the scheduler into one step loop. Each
``step`` advances every decode-ready request by exactly one token and advances each
not-yet-prefilled prompt by a bounded cached prefill chunk. A freshly admitted short
request can still emit its first token in one step; a long prompt may take several
steps to become decode-ready, letting already-running requests keep decoding between
chunks. Finished requests are freed at the end of the step (the decode-step boundary),
which returns their blocks and budget so a queued request can be admitted next step.

Token selection is **per request**: the fused decode forward stays shared (one
``decode_many`` over the whole batch), but each row of the resulting logits is sampled
under its own request's :class:`~llm_infer.serving.sampler.SamplingParams`, against that
request's own generated history, drawn from that request's own seeded generator. Greedy
requests (the default — temperature 0, the proven oracle path) take the vectorized argmax
fast-path. Because each request's draw depends only on its own seed and decode steps, a
sampled request produces the identical sequence run alone or batched with others, and a
greedy request stays token-for-token the proven greedy path.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from llm_infer.kv_cache.block_allocator import BlockPoolEvent, OutOfBlocksError
from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.qwen import QwenModel
from llm_infer.profiling import TimingProfiler
from llm_infer.scheduler.scheduler import (
    Scheduler,
    blocks_for_footprint,
    max_blocks_for,
)
from llm_infer.serving.request import Request
from llm_infer.serving.sampler import GREEDY, SamplingParams, sample_row
from llm_infer.serving.speculative import PromptLookupDraft, SpeculativeDecodingConfig
from llm_infer.tracing import FinishReason, TokenSource, TraceEvent, TraceEventName, TraceRecorder


@dataclass
class StepResult:
    """What happened in one engine step — enough to trace the vertical slice."""

    admitted: list[str] = field(default_factory=list)
    prefill_chunks: dict[str, tuple[int, int]] = field(default_factory=dict)
    finished: list[str] = field(default_factory=list)
    tokens: dict[str, list[int | torch.Tensor]] = field(default_factory=dict)


class InferenceEngine:
    """Runs generation for a set of requests over a shared paged KV-cache (greedy by default)."""

    def __init__(
        self,
        model: QwenModel,
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
    ) -> None:
        if prefill_chunk_size is not None and prefill_chunk_size < 1:
            raise ValueError(f"prefill_chunk_size must be >= 1 when set; got {prefill_chunk_size}")
        self.model = model
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
        # greedy (temperature 0 — token-for-token the proven oracle path); the rollout/benchmark
        # path passes one seeded temperature/top-p SamplingParams that every request inherits.
        self.default_sampling = default_sampling or GREEDY
        self.profiler = profiler
        self.model.profiler = profiler
        self.prefill_chunk_size = prefill_chunk_size
        self.speculative = PromptLookupDraft(speculative) if speculative is not None else None
        self.trace = trace
        if trace is not None:
            # Emit block_allocated/block_freed from the real physical boundary: the allocator
            # is the single owner of the free pool, so its events are honest for refcounted
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
        # Observability only: a running tally of preemptions for the serving metrics endpoint.
        # Read live at scrape time; it never influences scheduling or decoding.
        self.preemption_count = 0

    def add_request(self, request: Request) -> None:
        """Register and queue a request. Duplicate ids are rejected loudly."""
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request id {request.request_id!r}")
        self._requests[request.request_id] = request
        self.scheduler.add(request)

    def abort(self, request_id: str) -> bool:
        """Drop a request mid-flight, freeing its KV — for a client that disconnected.

        Handles both scheduler states honestly: a still-waiting request is removed from the
        queue (no KV yet), a running request has its block table freed at the allocator
        boundary and its budget released, exactly like the finish-sweep. Idempotent: a request
        already finished or unknown returns ``False``. Must be called between steps (the
        serving loop owns the engine on one thread), never mid-forward.
        """
        request = self._requests.pop(request_id, None)
        if request is None:
            return False
        if request in self.scheduler.waiting:
            self.scheduler.waiting.remove(request)
            return True
        if request in self.scheduler.running:
            if request.block_table is not None:
                request.block_table.free()
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
        result = StepResult()
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

            for request in [r for r in self.scheduler.running if r.finished]:
                self._trace_request_finished(request)
                request.block_table.free()
                self.scheduler.release(request)
            self._trace_batch_size_changed()
            self._trace_throughput_sample(result)

            return result
        finally:
            self._trace_step = None

    # --- preemption (recompute) ---------------------------------------------------------
    #
    # Trigger points: the two places a running request needs the allocator to hand out a
    # *new* physical block. Decode appends one token per step (a new block only when the
    # current one fills); a prefill chunk caches up to ``chunk_size`` new positions. Before
    # either forward we ensure the free pool can cover that growth, preempting victims until
    # it can. Reserving here — at the engine boundary, before the model touches the cache —
    # keeps recovery clean: no forward runs half-done, so a preempted-then-resumed request
    # rebuilds from a pristine state and stays token-exact.

    def _blocks_to_grow(self, request: Request, new_tokens: int) -> int:
        """How many *new* physical blocks ``request`` must pull to cache ``new_tokens`` more."""
        table = request.block_table
        current = table.num_blocks if table is not None else 0
        length = table.length if table is not None else 0
        needed = -(-(length + new_tokens) // self.scheduler.block_size)  # ceil division
        return max(0, needed - current)

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
        """Evict ``victim`` by recompute: free its KV honestly, keep its tokens, requeue it."""
        self.preemption_count += 1
        freed_blocks = victim.block_table.num_blocks if victim.block_table is not None else 0
        if victim.block_table is not None:
            victim.block_table.free()  # fires honest block_freed at the allocator boundary
        victim.reset_for_recompute()
        self.scheduler.requeue(victim)
        self._emit_trace(
            "request_preempted",
            request_id=victim.request_id,
            preempt_reason="kv_pressure",
            block_count=freed_blocks,
            generated_tokens=len(victim.generated),
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
        return any(candidate is request for candidate in self.scheduler.running)

    def _resume_requests(self, requests: list[Request], result: StepResult) -> None:
        """Rebuild each preempted request's KV by recompute before it decodes again.

        Recompute, not swap: the request kept its generated tokens through preemption, so a
        fresh prefill over ``recompute_prompt_ids`` reconstructs exactly the KV state it was
        evicted in. No token is sampled — the request already owns its generated ids — so on the
        next step it decodes its next token, identical to the uninterrupted run.
        """
        for request in requests:
            # A queued resume can be preempted again while making room for an earlier one this
            # step; check membership live (see _is_running) so a request evicted mid-loop is
            # skipped and re-picked once it is re-admitted, never resumed out of the waiting queue.
            if not self._is_running(request):
                continue
            self._recompute_prefill(request)
            self._emit_trace(
                "request_resumed",
                request_id=request.request_id,
                prompt_tokens=len(request.prompt_ids),
                generated_tokens=len(request.generated),
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
        if request.block_table is None:
            request.block_table = self.cache.new_request()
            request.block_table.owner = request.request_id

        total = len(sequence)
        chunk_size = self.prefill_chunk_size or total
        while request.prompt_cached_tokens < total:
            start = request.prompt_cached_tokens
            step = min(chunk_size, total - start)
            self._ensure_pool_room(request, step)
            self._emit_trace(
                "prefill_chunk_started",
                request_id=request.request_id,
                start_pos=start,
                end_pos=start + step,
                total_prompt_tokens=total,
            )
            with self._record_time("prefill"):
                self._write_recompute_chunk(request, sequence, start, step)
            request.prompt_cached_tokens = start + step
            self._emit_trace(
                "prefill_chunk_progress",
                request_id=request.request_id,
                start_pos=start,
                end_pos=start + step,
                cached_tokens=request.prompt_cached_tokens,
                total_prompt_tokens=total,
                completed=request.prompt_cached_tokens == total,
            )
        request.prefilled = True

    def _write_recompute_chunk(
        self, request: Request, sequence: list[int], start: int, count: int
    ) -> None:
        """Write KV for ``sequence[start:start+count]`` via the model's cached prefill path."""
        if start == 0 and count == len(sequence):
            self.model.prefill(sequence, self.cache, request.block_table)
            return
        prefill_chunk = getattr(self.model, "prefill_chunk", None)
        if prefill_chunk is None:
            raise TypeError(
                f"{type(self.model).__name__} must implement prefill_chunk() "
                "to resume a preempted request in chunks"
            )
        prefill_chunk(
            sequence,
            self.cache,
            request.block_table,
            start_pos=start,
            chunk_size=count,
        )

    def _reserved_blocks(self, request: Request) -> int:
        """Blocks the admit event reports reserved — footprint under preemption, else worst case."""
        if self.preemption:
            return blocks_for_footprint(request, self.scheduler.block_size)
        return max_blocks_for(request, self.scheduler.block_size)

    def _prefill_requests(self, requests: list[Request], result: StepResult) -> None:
        """Prefill unstarted requests, sharing prompt blocks for declared sibling groups."""
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
                for candidate in requests
                if candidate.prefix_group_id == request.prefix_group_id
                and (not self.preemption or self._is_running(candidate))
            ]
            if len(group) == 1:
                self._prefill_one(request, result)
                handled.add(request.request_id)
                continue
            self._prefill_shared_group(group, result)
            handled.update(candidate.request_id for candidate in group)

    def _prefill_one(self, request: Request, result: StepResult) -> None:
        logits = self._cache_prompt_chunk(request, result)
        if logits is None:
            return
        request.prefilled = True
        with self._record_time("sampling"):
            token = self._sample_one(logits, request)
        self._record(request, token, self._eos_flags(token.reshape(1), [request])[0], result)
        self._trace_decode_step([request], [token], token_source="prefill")

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

        leader.prefilled = True
        leader.prompt_cached_tokens = len(prompt_ids)
        for request in requests:
            if request is leader:
                continue
            request.block_table = self.cache.fork_request(leader.block_table)
            request.block_table.owner = request.request_id
            request.prompt_cached_tokens = leader.prompt_cached_tokens
            request.prefilled = True

        with self._record_time("sampling"):
            tokens = [self._sample_one(logits, request) for request in requests]
        eos_flags = self._eos_flags(torch.stack(tokens), requests)
        for request, token, is_eos in zip(requests, tokens, eos_flags, strict=True):
            self._record(request, token, is_eos, result)
        self._trace_decode_step(requests, tokens, token_source="prefill")

    def _decode_requests(self, requests: list[Request], result: StepResult) -> None:
        """Advance decode-ready requests, optionally using prompt-lookup speculation."""
        if self.preemption:
            requests = self._make_decode_room(requests)
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

    def _decode_normal(self, requests: list[Request], result: StepResult) -> None:
        """The original one-token batched decode path: shared forward, per-row sampling."""
        last_tokens = torch.stack([request.last_token_tensor for request in requests]).to(
            self.model.device
        )
        with self._record_time("decode"):
            logits = self.model.decode_many(
                self.cache,
                [request.block_table for request in requests],
                last_tokens,
            )
        with self._record_time("sampling"):
            tokens = self._sample_rows(logits, requests)
        eos_flags = self._eos_flags(torch.stack(tokens), requests)
        for request, token, is_eos in zip(requests, tokens, eos_flags, strict=True):
            self._record(request, token, is_eos, result)
        self._trace_decode_step(requests, list(tokens), token_source="decode")

    def _params_for(self, request: Request) -> SamplingParams:
        """The request's own sampling params, or the engine default when it set none."""
        return request.sampling if request.sampling is not GREEDY else self.default_sampling

    def _sample_one(self, logits: torch.Tensor, request: Request) -> torch.Tensor:
        """Sample one token from a 1-D ``(vocab,)`` row under this request (prefill path)."""
        return sample_row(
            logits, self._params_for(request), request.generated, request.generator(logits.device)
        )

    def _sample_rows(self, logits: torch.Tensor, requests: list[Request]) -> list[torch.Tensor]:
        """Sample one token per row of ``(B, vocab)`` logits, each under its own request.

        Greedy rows take the vectorized argmax (no RNG); the rest are sampled per row under that
        request's params, against its own generated history, from its own seeded generator — so a
        request's draw is independent of its batchmates. Returns scalar long tensors on device.
        """
        if logits.ndim != 2:
            raise ValueError(f"expected 2-D logits, got shape {tuple(logits.shape)}")
        params = [self._params_for(request) for request in requests]
        tokens: list[torch.Tensor] = [None] * len(requests)  # type: ignore[list-item]

        greedy_rows = [i for i, p in enumerate(params) if p.is_greedy]
        if greedy_rows:
            index = torch.tensor(greedy_rows, device=logits.device)
            argmax = torch.argmax(logits.index_select(0, index), dim=-1)
            for position, token in zip(greedy_rows, argmax, strict=True):
                tokens[position] = token

        for i, (request, p) in enumerate(zip(requests, params, strict=True)):
            if p.is_greedy:
                continue
            tokens[i] = sample_row(
                logits[i], p, request.generated, request.generator(logits.device)
            )
        return tokens

    def _make_decode_room(self, requests: list[Request]) -> list[Request]:
        """Ensure the whole decode batch can grow by one token; preempt LIFO victims if not.

        ``decode_many`` allocates for every surviving member in one call, so room must cover the
        batch's *total* growth, not one request at a time. Each decode step appends exactly one
        token (at most one new block per request). A victim is the most-recently-admitted running
        request and may itself be in this batch — preempting it drops it from the step and lowers
        the demand. We preempt until the free pool covers the survivors' combined growth.

        Forward progress holds: the block-needers are the batch members, and preempting strictly
        shrinks the batch, so the demand reaches zero before victims run out.
        """
        survivors = [r for r in requests if r.prefilled and r.block_table is not None]
        while True:
            survivors = [r for r in survivors if r.prefilled and r.block_table is not None]
            demand = sum(self._blocks_to_grow(r, 1) for r in survivors)
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
        decode_tokens = getattr(self.model, "decode_tokens", None)
        if decode_tokens is None:
            raise TypeError(
                f"{type(self.model).__name__} must implement decode_tokens() "
                "when speculative decoding is enabled"
            )
        if request.block_table is None:
            raise ValueError(f"request {request.request_id!r} has no block table")

        original_length = request.block_table.length
        draft_tensor = torch.tensor(draft, dtype=torch.long, device=self.model.device)
        verify_input = torch.cat(
            [
                request.last_token_tensor.to(self.model.device).reshape(1),
                draft_tensor,
            ]
        )
        with self._record_time("speculative_decode"):
            logits = decode_tokens(self.cache, request.block_table, verify_input)

        verifier_tokens = torch.argmax(logits, dim=-1)
        accepted = self._accepted_prefix_length(verifier_tokens[:-1], draft_tensor)

        emitted: list[int | torch.Tensor] = []
        emitted.extend(draft[:accepted])
        if not self._contains_eos(emitted, request):
            if accepted == len(draft):
                emitted.append(verifier_tokens[-1])
            else:
                emitted.append(verifier_tokens[accepted])

        emitted = self._truncate_after_eos(emitted, request)
        if not emitted:
            raise ValueError("speculative verification produced no token to emit")

        request.block_table.length = min(original_length + len(emitted), request.block_table.length)
        for token in emitted:
            if request.finished:
                break
            self._record(request, token, self._is_eos(token, request), result)
        self._trace_decode_step([request], emitted, token_source="speculative")

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

    def _cache_prompt_chunk(self, request: Request, result: StepResult) -> torch.Tensor | None:
        """Cache one prompt chunk and return final-prompt logits when ready to sample."""
        if request.block_table is None:
            request.block_table = self.cache.new_request()
            request.block_table.owner = request.request_id

        start_pos = request.prompt_cached_tokens
        if request.block_table.length != start_pos:
            raise ValueError(
                f"request {request.request_id!r} block table length "
                f"{request.block_table.length} != cached prompt length {start_pos}"
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
        self._emit_trace(
            "prefill_chunk_started",
            request_id=request.request_id,
            start_pos=start_pos,
            end_pos=end_pos,
            total_prompt_tokens=len(request.prompt_ids),
        )

        with self._record_time("prefill"):
            if start_pos == 0 and end_pos == len(request.prompt_ids):
                logits = self.model.prefill(request.prompt_ids, self.cache, request.block_table)
            else:
                prefill_chunk = getattr(self.model, "prefill_chunk", None)
                if prefill_chunk is None:
                    raise TypeError(
                        f"{type(self.model).__name__} must implement prefill_chunk() "
                        "when prefill_chunk_size splits a prompt"
                    )
                logits = prefill_chunk(
                    request.prompt_ids,
                    self.cache,
                    request.block_table,
                    start_pos=start_pos,
                    chunk_size=chunk_size,
                )
        request.prompt_cached_tokens = end_pos
        self._emit_trace(
            "prefill_chunk_progress",
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

    def _record(
        self, request: Request, token: int | torch.Tensor, is_eos: bool, result: StepResult
    ) -> None:
        """Append a sampled token to a request and note it (and any finish) in the step result."""
        request.record(token, is_eos=is_eos)
        result.tokens.setdefault(request.request_id, []).append(token)
        if request.finished and request.request_id not in result.finished:
            result.finished.append(request.request_id)

    def run(self) -> dict[str, list[int]]:
        """Step until the queue and running set drain; return each request's generated ids."""
        with self._record_time("total_wall"):
            while self.scheduler.has_work():
                self.step()
        with self._record_host_time("cpu_gpu_sync"):
            return {rid: request.generated for rid, request in self._requests.items()}

    def _eos_flags(self, tokens: torch.Tensor, requests: list[Request]) -> list[bool]:
        """Return per-request EOS flags, using one host sync for the common EOS-set case."""
        flat = tokens.reshape(-1)
        if len(flat) != len(requests):
            raise ValueError(f"token/request count mismatch: {len(flat)} vs {len(requests)}")
        eos_sets = {request.eos_token_ids for request in requests}
        if len(eos_sets) == 1:
            eos = torch.tensor(
                sorted(next(iter(eos_sets))),
                dtype=torch.long,
                device=flat.device,
            )
            mask = (flat.unsqueeze(-1) == eos).any(dim=-1)
            with self._record_host_time("cpu_gpu_sync"):
                return [bool(flag) for flag in mask.cpu().tolist()]

        flags: list[bool] = []
        with self._record_host_time("cpu_gpu_sync"):
            for token, request in zip(flat, requests, strict=True):
                flags.append(int(token.cpu().item()) in request.eos_token_ids)
        return flags

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

    def _trace_decode_step(
        self,
        requests: list[Request],
        tokens: list[int | torch.Tensor],
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
            generated_tokens=len(request.generated),
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

    def _emit_trace(self, event: TraceEventName, **fields: object) -> None:
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
                **fields,
            )
        )

    def _trace_token_id(self, token: int | torch.Tensor) -> int:
        if isinstance(token, torch.Tensor):
            with self._record_host_time("cpu_gpu_sync"):
                return int(token.cpu().item())
        return token

    def _record_time(self, name: str):
        if self.profiler is None:
            return _NullTimer()
        return self.profiler.record(name)

    def _record_host_time(self, name: str):
        if self.profiler is None:
            return _NullTimer()
        return self.profiler.host(name)


class _NullTimer:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None
