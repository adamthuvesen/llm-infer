"""Decode, sampling, and finish handling for the continuous-batching decode loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llm_infer.kv_cache.block_allocator import OutOfBlocksError
from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.serving.request import Request
from llm_infer.serving.sampler import GREEDY, SamplingParams, sample_row

if TYPE_CHECKING:
    from llm_infer.serving.engine import StepResult
    from llm_infer.serving.engine_contract import EngineMixinHost
else:
    EngineMixinHost = object


class EngineDecodeMixin(EngineMixinHost):
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
        self._release_finished_in(requests, result)

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
        tokens: list[torch.Tensor | None] = [None] * len(requests)

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
        if any(token is None for token in tokens):
            raise RuntimeError("sampled fewer tokens than requests")
        filled: list[torch.Tensor] = []
        for token in tokens:
            if token is None:
                raise RuntimeError("internal sampling gap")
            filled.append(token)
        return filled

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
        self, request: Request, token: int | torch.Tensor, is_eos: bool, result: StepResult
    ) -> None:
        """Append a sampled token to a request and note it (and any finish) in the step result."""
        request.record(token, is_eos=is_eos)
        result.tokens.setdefault(request.request_id, []).append(token)
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
