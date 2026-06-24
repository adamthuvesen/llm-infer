"""The minimal continuous-batching decode loop — the Phase B vertical slice runner.

Wires the model, the paged KV-cache, and the scheduler into one step loop. Each
``step`` advances every decode-ready request by exactly one token and advances each
not-yet-prefilled prompt by a bounded cached prefill chunk. A freshly admitted short
request can still emit its first token in one step; a long prompt may take several
steps to become decode-ready, letting already-running requests keep decoding between
chunks. Finished requests are freed at the end of the step (the decode-step boundary),
which returns their blocks and budget so a queued request can be admitted next step.

Token selection is pluggable through a :class:`~llm_infer.serving.sampler.Sampler`; it
defaults to greedy (temperature 0 — the proven oracle path) and the rlvr-sql rollout passes
a seeded temperature/top-p sampler. There is still no streaming or OpenAI-compatible serving
surface. Under greedy, each request runs through the same cached path, so batching two
requests gives token-for-token the same result as running each alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.qwen import QwenModel
from llm_infer.profiling import TimingProfiler
from llm_infer.scheduler.scheduler import Scheduler, max_blocks_for
from llm_infer.serving.request import Request
from llm_infer.serving.sampler import Sampler
from llm_infer.serving.speculative import PromptLookupDraft, SpeculativeDecodingConfig
from llm_infer.tracing import FinishReason, TraceEvent, TraceEventName, TraceRecorder


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
        sampler: Sampler | None = None,
        profiler: TimingProfiler | None = None,
        prefill_chunk_size: int | None = None,
        speculative: SpeculativeDecodingConfig | None = None,
        trace: TraceRecorder | None = None,
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
        self.scheduler = Scheduler(num_blocks, block_size)
        # Default to greedy (temperature 0) — token-for-token the proven oracle path. The
        # rollout passes Sampler(temperature=1.0, top_p=1.0, seed=...) for sampled decoding.
        self.sampler = sampler or Sampler()
        if speculative is not None and not self.sampler.is_greedy:
            raise ValueError("speculative decoding v1 supports only greedy sampling")
        self.profiler = profiler
        self.model.profiler = profiler
        self.prefill_chunk_size = prefill_chunk_size
        self.speculative = PromptLookupDraft(speculative) if speculative is not None else None
        self.trace = trace
        self._step_index = 0
        self._trace_sequence = 0
        self._trace_step: int | None = None
        self._trace_start_time = time.perf_counter()
        self._trace_total_tokens = 0
        self._last_traced_batch_size = 0
        self._requests: dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        """Register and queue a request. Duplicate ids are rejected loudly."""
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request id {request.request_id!r}")
        self._requests[request.request_id] = request
        self.scheduler.add(request)

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
            for request in self.scheduler.admit():
                result.admitted.append(request.request_id)
                self._emit_trace(
                    "request_admitted",
                    request_id=request.request_id,
                    prompt_tokens=len(request.prompt_ids),
                    max_new_tokens=request.max_new_tokens,
                    prefix_group_id=request.prefix_group_id,
                    reserved_blocks=max_blocks_for(request, self.scheduler.block_size),
                )
            self._trace_batch_size_changed()

            to_decode: list[Request] = []
            to_prefill: list[Request] = []
            for request in self.scheduler.running:
                if request.prefilled:
                    to_decode.append(request)
                    continue
                to_prefill.append(request)

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

    def _prefill_requests(self, requests: list[Request], result: StepResult) -> None:
        """Prefill unstarted requests, sharing prompt blocks for declared sibling groups."""
        handled: set[str] = set()
        for request in requests:
            if request.request_id in handled:
                continue
            if request.prefix_group_id is None:
                self._prefill_one(request, result)
                handled.add(request.request_id)
                continue

            group = [
                candidate
                for candidate in requests
                if candidate.prefix_group_id == request.prefix_group_id
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
            token = self.sampler.sample(logits)
        self._record(request, token, self._eos_flags(token.reshape(1), [request])[0], result)

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
            request.prompt_cached_tokens = leader.prompt_cached_tokens
            request.prefilled = True

        with self._record_time("sampling"):
            tokens = [self.sampler.sample(logits) for _ in requests]
        eos_flags = self._eos_flags(torch.stack(tokens), requests)
        for request, token, is_eos in zip(requests, tokens, eos_flags, strict=True):
            self._record(request, token, is_eos, result)

    def _decode_requests(self, requests: list[Request], result: StepResult) -> None:
        """Advance decode-ready requests, optionally using prompt-lookup speculation."""
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
        """The original one-token batched decode path."""
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
            tokens = self.sampler.sample_many(logits)
        eos_flags = self._eos_flags(tokens, requests)
        for request, token, is_eos in zip(requests, tokens, eos_flags, strict=True):
            self._record(request, token, is_eos, result)
        self._trace_decode_step(requests, [token for token in tokens.reshape(-1)])

    def _draft_for(self, request: Request) -> list[int]:
        """Return a draft only when there is room for draft tokens plus verifier recovery."""
        if self.speculative is None:
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
        self._trace_decode_step([request], emitted)

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

    def _trace_decode_step(self, requests: list[Request], tokens: list[int | torch.Tensor]) -> None:
        if self.trace is None:
            return
        self._emit_trace(
            "decode_step",
            request_ids=tuple(request.request_id for request in requests),
            batch_size=len(requests),
            token_ids=tuple(self._trace_token_id(token) for token in tokens),
            tokens_emitted=len(tokens),
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
        elapsed = max(time.perf_counter() - self._trace_start_time, 1e-12)
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
