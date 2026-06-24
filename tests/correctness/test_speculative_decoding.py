"""Speculative decoding v1: prompt-lookup draft, greedy verifier, safe fallback."""

from __future__ import annotations

import torch

from llm_infer.kv_cache import BlockTable, PagedKVCache
from llm_infer.serving import (
    InferenceEngine,
    PromptLookupDraft,
    Request,
    SamplingParams,
    SpeculativeDecodingConfig,
)


class ScriptedToyModel:
    """Model-shaped object whose greedy path follows a per-prompt token script."""

    def __init__(self, scripts: dict[tuple[int, ...], list[int]]) -> None:
        self.scripts = scripts
        self.num_layers = 1
        self.num_kv_heads = 1
        self.head_dim = 2
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.profiler = None
        self.decode_tokens_calls = 0
        self._states: dict[int, _ToyState] = {}

    def prefill(
        self, prompt_ids: list[int], cache: PagedKVCache, table: BlockTable
    ) -> torch.Tensor:
        table.reserve(len(prompt_ids))
        self._write(cache, table, start_pos=0, count=len(prompt_ids))
        table.length = len(prompt_ids)
        script = self.scripts[tuple(prompt_ids)]
        self._states[id(table)] = _ToyState(prompt_ids=list(prompt_ids), script=script)
        return self._logits(script[0])

    def prefill_chunk(
        self,
        prompt_ids: list[int],
        cache: PagedKVCache,
        table: BlockTable,
        *,
        start_pos: int,
        chunk_size: int,
    ) -> torch.Tensor:
        if start_pos == 0:
            self._states[id(table)] = _ToyState(
                prompt_ids=list(prompt_ids),
                script=self.scripts[tuple(prompt_ids)],
            )
        end_pos = min(len(prompt_ids), start_pos + chunk_size)
        table.reserve(end_pos - start_pos)
        self._write(cache, table, start_pos=start_pos, count=end_pos - start_pos)
        table.length = end_pos
        return self._logits(self.scripts[tuple(prompt_ids)][0])

    def decode_many(
        self,
        cache: PagedKVCache,
        tables: list[BlockTable],
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.as_tensor(token_ids, dtype=torch.long)
        logits: list[torch.Tensor] = []
        for table, token in zip(tables, tokens.tolist(), strict=True):
            state = self._sync(table)
            table.reserve(1)
            self._write(cache, table, start_pos=table.length, count=1)
            table.length += 1
            state.cached_generated += 1
            state.cached_tokens.append(token)
            logits.append(self._logits(state.next_token()))
        return torch.stack(logits)

    def decode_tokens(
        self,
        cache: PagedKVCache,
        table: BlockTable,
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        self.decode_tokens_calls += 1
        state = self._sync(table)
        tokens = torch.as_tensor(token_ids, dtype=torch.long).tolist()
        table.reserve(len(tokens))
        self._write(cache, table, start_pos=table.length, count=len(tokens))
        table.length += len(tokens)

        logits: list[torch.Tensor] = []
        for token in tokens:
            state.cached_generated += 1
            state.cached_tokens.append(token)
            logits.append(self._logits(state.next_token()))
        return torch.stack(logits)

    def _sync(self, table: BlockTable) -> _ToyState:
        state = self._states[id(table)]
        prompt_len = len(state.prompt_ids)
        state.cached_generated = table.length - prompt_len
        state.cached_tokens = state.cached_tokens[: table.length]
        return state

    def _write(self, cache: PagedKVCache, table: BlockTable, *, start_pos: int, count: int) -> None:
        rows = torch.arange(start_pos, start_pos + count, dtype=torch.float32).reshape(-1, 1, 1)
        key = torch.cat([rows, rows + 0.5], dim=-1)
        cache.write(table, layer=0, start_pos=start_pos, key=key, value=key + 100.0)

    def _logits(self, token_id: int) -> torch.Tensor:
        logits = torch.full((64,), -100.0)
        logits[token_id] = 100.0
        return logits


class _ToyState:
    def __init__(self, *, prompt_ids: list[int], script: list[int]) -> None:
        self.prompt_ids = prompt_ids
        self.script = script
        self.cached_generated = 0
        self.cached_tokens = list(prompt_ids)

    def next_token(self) -> int:
        if self.cached_generated >= len(self.script):
            return self.script[-1]
        return self.script[self.cached_generated]


class NoSpecVerifierToyModel:
    """Small prefix/chunk model that fails if speculative verification is called."""

    def __init__(self) -> None:
        self.num_layers = 1
        self.num_kv_heads = 1
        self.head_dim = 2
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.profiler = None

    def prefill(
        self, prompt_ids: list[int], cache: PagedKVCache, table: BlockTable
    ) -> torch.Tensor:
        return self.prefill_chunk(prompt_ids, cache, table, start_pos=0, chunk_size=len(prompt_ids))

    def prefill_chunk(
        self,
        prompt_ids: list[int],
        cache: PagedKVCache,
        table: BlockTable,
        *,
        start_pos: int,
        chunk_size: int,
    ) -> torch.Tensor:
        end_pos = min(len(prompt_ids), start_pos + chunk_size)
        table.reserve(end_pos - start_pos)
        rows = torch.arange(start_pos, end_pos, dtype=torch.float32).reshape(-1, 1, 1)
        key = torch.cat([rows, rows + 0.5], dim=-1)
        cache.write(table, layer=0, start_pos=start_pos, key=key, value=key + 10.0)
        table.length = end_pos
        return self._logits(7)

    def decode_many(
        self,
        cache: PagedKVCache,
        tables: list[BlockTable],
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.as_tensor(token_ids, dtype=torch.long)
        positions = [table.length for table in tables]
        for table in tables:
            table.reserve(1)
        key = torch.stack(
            [
                torch.tensor([[float(pos), float(token)]], dtype=torch.float32)
                for pos, token in zip(positions, tokens.tolist(), strict=True)
            ]
        )
        cache.write_many(tables, layer=0, positions=positions, key=key, value=key + 20.0)
        for table, pos in zip(tables, positions, strict=True):
            table.length = pos + 1
        return torch.stack([self._logits(int(token) + 1) for token in tokens.tolist()])

    def decode_tokens(self, *args: object) -> torch.Tensor:
        raise AssertionError("decode_tokens must not run when speculative decoding is off")

    def _logits(self, token_id: int) -> torch.Tensor:
        logits = torch.full((32,), -100.0)
        logits[token_id] = 100.0
        return logits


def _run(
    *,
    prompt: list[int],
    script: list[int],
    eos: frozenset[int] = frozenset({63}),
    speculative: bool,
) -> tuple[list[int], ScriptedToyModel]:
    model = ScriptedToyModel({tuple(prompt): script})
    engine = InferenceEngine(
        model,
        block_size=4,
        num_blocks=8,
        speculative=SpeculativeDecodingConfig(max_draft_tokens=2, max_ngram_size=3)
        if speculative
        else None,
    )
    engine.add_request(Request("r", prompt, len(script), eos))
    return engine.run()["r"], model


def test_prompt_lookup_uses_max_draft_length() -> None:
    draft = PromptLookupDraft(SpeculativeDecodingConfig(max_draft_tokens=2, max_ngram_size=4))

    assert draft.draft([1, 2, 3, 4, 1, 2], max_tokens=3) == [3, 4]
    assert draft.draft([1, 2, 3, 4, 1, 2], max_tokens=1) == [3]


def test_full_draft_acceptance_matches_normal_greedy_path() -> None:
    prompt = [1, 2, 3, 1, 2]
    script = [3, 1, 2, 9]

    baseline, _ = _run(prompt=prompt, script=script, speculative=False)
    speculative, model = _run(prompt=prompt, script=script, speculative=True)

    assert speculative == baseline == script
    assert model.decode_tokens_calls == 1


def test_partial_draft_acceptance_emits_verifier_fallback() -> None:
    prompt = [1, 2, 3, 1, 2]
    script = [3, 1, 9, 10]

    baseline, _ = _run(prompt=prompt, script=script, speculative=False)
    speculative, model = _run(prompt=prompt, script=script, speculative=True)

    assert speculative == baseline == script
    assert model.decode_tokens_calls == 1


def test_rejected_draft_falls_back_to_normal_next_token() -> None:
    prompt = [1, 2, 3, 1, 2]
    script = [3, 8, 9]

    baseline, _ = _run(prompt=prompt, script=script, speculative=False)
    speculative, model = _run(prompt=prompt, script=script, speculative=True)

    assert speculative == baseline == script
    assert model.decode_tokens_calls == 1


def test_no_draft_uses_normal_decode_path() -> None:
    prompt = [10, 11, 12]
    script = [13, 14, 15]

    baseline, _ = _run(prompt=prompt, script=script, speculative=False)
    speculative, model = _run(prompt=prompt, script=script, speculative=True)

    assert speculative == baseline == script
    assert model.decode_tokens_calls == 0


def test_eos_in_accepted_draft_stops_before_extra_verifier_token() -> None:
    prompt = [1, 2, 3, 1, 2]
    script = [3, 1, 2, 9]

    output, model = _run(prompt=prompt, script=script, eos=frozenset({2}), speculative=True)

    assert output == [3, 1, 2]
    assert model.decode_tokens_calls == 1


def test_sampled_request_skips_the_speculative_path() -> None:
    """A non-greedy request never takes the greedy-verifier speculative path; it samples per-row.

    The guard is per request, not engine-wide: with speculation enabled and the engine default
    sampling (temperature > 0), the verifier (`decode_tokens`) is never called. The toy logits are
    one-hot, so the sampled draw still follows the script — output equals the greedy baseline.
    """
    prompt = [1, 2, 3, 1, 2]
    script = [3, 1, 2, 9]
    baseline, _ = _run(prompt=prompt, script=script, speculative=False)

    model = ScriptedToyModel({tuple(prompt): script})
    engine = InferenceEngine(
        model,
        block_size=4,
        num_blocks=8,
        default_sampling=SamplingParams(temperature=1.0),
        speculative=SpeculativeDecodingConfig(max_draft_tokens=2, max_ngram_size=3),
    )
    engine.add_request(Request("r", prompt, len(script), frozenset({63})))
    output = engine.run()["r"]

    assert output == baseline == script
    assert model.decode_tokens_calls == 0


def test_greedy_request_still_speculates_under_a_sampled_default() -> None:
    """An explicitly-greedy request still takes the speculative path even if the default samples."""
    prompt = [1, 2, 3, 1, 2]
    script = [3, 1, 2, 9]

    model = ScriptedToyModel({tuple(prompt): script})
    engine = InferenceEngine(
        model,
        block_size=4,
        num_blocks=8,
        default_sampling=SamplingParams(temperature=1.0),
        speculative=SpeculativeDecodingConfig(max_draft_tokens=2, max_ngram_size=3),
    )
    engine.add_request(
        Request("r", prompt, len(script), frozenset({63}), sampling=SamplingParams(temperature=0.0))
    )
    output = engine.run()["r"]

    assert output == script
    assert model.decode_tokens_calls == 1


def test_default_chunked_prefix_caching_does_not_call_speculative_verifier() -> None:
    model = NoSpecVerifierToyModel()
    engine = InferenceEngine(model, block_size=4, num_blocks=24, prefill_chunk_size=2)
    requests = [
        Request(f"p0-g{idx}", [11, 12, 13, 14, 15, 16], 3, frozenset({31}), "p0")
        for idx in range(4)
    ]
    for request in requests:
        engine.add_request(request)

    output = engine.run()

    assert output == {request.request_id: [7, 8, 9] for request in requests}
