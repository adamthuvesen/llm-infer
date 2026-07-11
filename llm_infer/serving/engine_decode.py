"""Decode, sampling, and finish handling for the continuous-batching decode loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from llm_infer.kv_cache.block_allocator import OutOfBlocksError
from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.serving.request import Request
from llm_infer.serving.sampler import GREEDY, SamplingParams, sample_row, sample_rows

if TYPE_CHECKING:
    from llm_infer.model.decode_plan import DecodeWindowPlan
    from llm_infer.serving.engine import StepResult
    from llm_infer.serving.engine_contract import EngineMixinHost
else:
    EngineMixinHost = object


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


class EngineDecodeMixin(EngineMixinHost):
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
        """The planned window can skip copy-on-write only when every live block is private."""
        for request in requests:
            table = request.block_table
            if table is None:
                return False
            if request.prefix_group_id is None:
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
        window = self._decode_window
        ids = tuple(request.request_id for request in requests)
        if window is not None and window.request_ids != ids:
            self._drain_decode_window(result)
            window = None
            requests = [request for request in requests if not request.finished]
            if not requests:
                return
            ids = tuple(request.request_id for request in requests)
        if window is None:
            pending = self._pending_flush
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
                window.plan = self.model.open_decode_window(
                    self.cache,
                    [request.block_table for request in requests],
                    window.budget,
                )
            self._decode_window = window

        if window.pending:
            last_tokens = window.pending[-1]
        else:
            pending = self._pending_flush
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
                    logits = self.model.decode_window_step(self.cache, window.plan, last_tokens)
            else:
                logits = self.model.decode_many(
                    self.cache,
                    [request.block_table for request in requests],
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
        window = self._decode_window
        self._decode_window = None
        previous = self._pending_flush
        if window is not None and window.pending:
            self._pending_flush = self._stage_window_flush(window)
        if previous is not None and previous is not self._pending_flush:
            self._consume_window_flush(previous, result)

    def _drain_decode_window(self, result: StepResult) -> None:
        """Flush the open window and consume every staged copy — request state is current after.

        The synchronous companion to the pipelined flush, for boundaries that need recorded
        state now: batch-composition changes, aborts, and leaving the window path.
        """
        self._flush_decode_window(result)
        pending = self._pending_flush
        self._pending_flush = None
        if pending is not None:
            self._consume_window_flush(pending, result)

    def _stage_window_flush(self, window: DecodeWindow) -> PendingWindowFlush:
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
        survivors = [r for r in requests if r.prefilled and r.block_table is not None]
        while True:
            survivors = [r for r in survivors if r.prefilled and r.block_table is not None]
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
            logits = self.model.decode_tokens(self.cache, request.block_table, verify_input)

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

        request.block_table.length = min(original_length + len(emitted), request.block_table.length)
        for token in emitted:
            if request.finished:
                break
            self._record(request, token, self._is_eos(token, request), result)
        self._trace_decode_step([request], emitted, token_source="speculative")
        # Verification reserved blocks for the whole draft; a partly-rejected draft rolled the
        # length back, so return the unused trailing blocks to the pool (a finisher's table is
        # freed wholesale by _release_finished_in, so only trim a still-running request).
        if not request.finished:
            request.block_table.trim_to_length()
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
                    self._free_block_table(request.block_table)
                self.scheduler.release(request)
                self._requests.pop(request.request_id, None)

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
