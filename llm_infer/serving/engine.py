"""The minimal continuous-batching decode loop — the Phase B vertical slice runner.

Wires the model, the paged KV-cache, and the scheduler into one step loop. Each
``step`` advances every running request by exactly one token: a freshly admitted
request emits its first token via cached prefill, an already-running one via a cached
decode. Finished requests are freed at the end of the step (the decode-step boundary),
which returns their blocks and budget so a queued request can be admitted next step.

This is the v1 loop only: greedy sampling, no streaming, no chunked prefill, no mixed
prefill/decode fusion. Each request runs through the same single-request cached path,
so batching two requests gives token-for-token the same result as running each alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.qwen import QwenModel
from llm_infer.scheduler.scheduler import Scheduler
from llm_infer.serving.request import Request
from llm_infer.serving.sampler import greedy


@dataclass
class StepResult:
    """What happened in one engine step — enough to trace the vertical slice."""

    admitted: list[str] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    tokens: dict[str, int] = field(default_factory=dict)


class InferenceEngine:
    """Runs greedy generation for a set of requests over a shared paged KV-cache."""

    def __init__(
        self,
        model: QwenModel,
        *,
        block_size: int,
        num_blocks: int,
        device: str = "cpu",
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
            request.block_table = self.cache.new_request()
            result.admitted.append(request.request_id)

        to_decode: list[Request] = []
        for request in self.scheduler.running:
            if request.prefilled:
                to_decode.append(request)
                continue
            logits = self.model.prefill(request.prompt_ids, self.cache, request.block_table)
            request.prefilled = True
            self._record(request, greedy(logits), result)

        if to_decode:
            logits = self.model.decode_many(
                self.cache,
                [r.block_table for r in to_decode],
                [r.last_token for r in to_decode],
            )
            for i, request in enumerate(to_decode):
                self._record(request, greedy(logits[i]), result)

        for request in [r for r in self.scheduler.running if r.finished]:
            request.block_table.free()
            self.scheduler.release(request)

        return result

    @staticmethod
    def _record(request: Request, token: int, result: StepResult) -> None:
        """Append a sampled token to a request and note it (and any finish) in the step result."""
        request.record(token)
        result.tokens[request.request_id] = token
        if request.finished:
            result.finished.append(request.request_id)

    def run(self) -> dict[str, list[int]]:
        """Step until the queue and running set drain; return each request's generated ids."""
        while self.scheduler.has_work():
            self.step()
        return {rid: request.generated for rid, request in self._requests.items()}
