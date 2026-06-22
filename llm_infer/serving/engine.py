"""The minimal continuous-batching decode loop — the Phase B vertical slice runner.

Wires the model, the paged KV-cache, and the scheduler into one step loop. Each
``step`` advances every running request by exactly one token: a freshly admitted
request emits its first token via cached prefill, an already-running one via a cached
decode. Finished requests are freed at the end of the step (the decode-step boundary),
which returns their blocks and budget so a queued request can be admitted next step.

Token selection is pluggable through a :class:`~llm_infer.serving.sampler.Sampler`; it
defaults to greedy (temperature 0 — the proven oracle path) and the rlvr-sql rollout passes
a seeded temperature/top-p sampler. Otherwise this is the v1 loop only: no streaming, no
chunked prefill, no mixed prefill/decode fusion. Under greedy, each request runs through the
same single-request cached path, so batching two requests gives token-for-token the same
result as running each alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.qwen import QwenModel
from llm_infer.profiling import TimingProfiler
from llm_infer.scheduler.scheduler import Scheduler
from llm_infer.serving.request import Request
from llm_infer.serving.sampler import Sampler


@dataclass
class StepResult:
    """What happened in one engine step — enough to trace the vertical slice."""

    admitted: list[str] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    tokens: dict[str, int | torch.Tensor] = field(default_factory=dict)


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
    ) -> None:
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
        self.profiler = profiler
        self.model.profiler = profiler
        self._requests: dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        """Register and queue a request. Duplicate ids are rejected loudly."""
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request id {request.request_id!r}")
        self._requests[request.request_id] = request
        self.scheduler.add(request)

    def step(self) -> StepResult:
        """Admit, advance every running request by one token, then free finished ones.

        Newly-admitted requests emit their first token via a (per-request) cached prefill;
        every already-running request advances by one token through a single **batched**
        decode forward (``decode_many``) rather than one forward each — the fused-batch decode
        that makes continuous batching a throughput win, not just a scheduling one.
        """
        result = StepResult()

        for request in self.scheduler.admit():
            result.admitted.append(request.request_id)

        to_decode: list[Request] = []
        to_prefill: list[Request] = []
        for request in self.scheduler.running:
            if request.prefilled:
                to_decode.append(request)
                continue
            to_prefill.append(request)

        self._prefill_requests(to_prefill, result)

        if to_decode:
            last_tokens = torch.stack([r.last_token_tensor for r in to_decode]).to(
                self.model.device
            )
            with self._record_time("decode"):
                logits = self.model.decode_many(
                    self.cache,
                    [r.block_table for r in to_decode],
                    last_tokens,
                )
            with self._record_time("sampling"):
                tokens = self.sampler.sample_many(logits)
            eos_flags = self._eos_flags(tokens, to_decode)
            for request, token, is_eos in zip(to_decode, tokens, eos_flags, strict=True):
                self._record(request, token, is_eos, result)

        for request in [r for r in self.scheduler.running if r.finished]:
            request.block_table.free()
            self.scheduler.release(request)

        return result

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
        request.block_table = self.cache.new_request()
        with self._record_time("prefill"):
            logits = self.model.prefill(request.prompt_ids, self.cache, request.block_table)
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

        leader = requests[0]
        leader.block_table = self.cache.new_request()
        with self._record_time("prefill"):
            logits = self.model.prefill(prompt_ids, self.cache, leader.block_table)
        leader.prefilled = True
        for request in requests[1:]:
            request.block_table = self.cache.fork_request(leader.block_table)
            request.prefilled = True

        with self._record_time("sampling"):
            tokens = [self.sampler.sample(logits) for _ in requests]
        eos_flags = self._eos_flags(torch.stack(tokens), requests)
        for request, token, is_eos in zip(requests, tokens, eos_flags, strict=True):
            self._record(request, token, is_eos, result)

    def _record(
        self, request: Request, token: int | torch.Tensor, is_eos: bool, result: StepResult
    ) -> None:
        """Append a sampled token to a request and note it (and any finish) in the step result."""
        request.record(token, is_eos=is_eos)
        result.tokens[request.request_id] = token
        if request.finished:
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
