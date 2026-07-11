"""CPU coverage for the optional native-paged decode backend path."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle
from llm_infer.kernels.torch_naive import TorchNaiveAttention
from llm_infer.kv_cache.paged_kv_cache import KVPagePlan, PagedKVCache
from llm_infer.model.layers import expand_grouped_kv
from llm_infer.model.runtime import load_model_runtime

_PROMPTS = ([1, 4, 7], [2, 5, 3, 6])


class PageTableReferenceAttention:
    """Reference backend that reads K/V through ``KVPagePlan`` instead of packed gather."""

    def __init__(self) -> None:
        self.reference = TorchNaiveAttention()
        self.page_plan: KVPagePlan | None = None
        self.num_qo_heads = 0
        self.num_kv_heads = 0
        self.head_dim = 0
        self.plan_calls = 0
        self.paged_calls = 0
        self.packed_calls = 0

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return self.reference.forward(query, key, value)

    def forward_decode_batch_packed(
        self,
        queries: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        self.packed_calls += 1
        return self.reference.forward_decode_batch_packed(
            queries, key, value, cu_seqlens_k, max_seqlen_k
        )

    def plan_decode_batch_paged(
        self,
        page_plan: KVPagePlan,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        del dtype
        self.page_plan = page_plan
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.plan_calls += 1

    def forward_decode_batch_paged(
        self, queries: torch.Tensor, paged_kv_cache: torch.Tensor
    ) -> torch.Tensor:
        self.paged_calls += 1
        if self.page_plan is None:
            raise RuntimeError("page plan missing")

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        lengths: list[int] = []
        indptr = self.page_plan.indptr.cpu().tolist()
        last_page_len = self.page_plan.last_page_len.cpu().tolist()
        for row, (start, end) in enumerate(zip(indptr[:-1], indptr[1:], strict=True)):
            page_count = end - start
            length = (page_count - 1) * self.page_plan.page_size + last_page_len[row]
            pages = self.page_plan.indices[start:end].to(torch.long)
            rows = paged_kv_cache.index_select(0, pages)
            keys.append(rows[:, 0].reshape(-1, self.num_kv_heads, self.head_dim)[:length])
            values.append(rows[:, 1].reshape(-1, self.num_kv_heads, self.head_dim)[:length])
            lengths.append(length)

        key_exp, value_exp = expand_grouped_kv(
            torch.cat(keys, dim=0),
            torch.cat(values, dim=0),
            num_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_axis=1,
        )
        cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=queries.device)
        cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32, device=queries.device).cumsum(0)
        return self.reference.forward_decode_batch_packed(
            queries, key_exp, value_exp, cu_seqlens, max(lengths)
        )


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    return write_tiny_pretrain_bundle(tmp_path)


def _prefill(model, cache: PagedKVCache) -> tuple[list, torch.Tensor]:
    tables, tokens = [], []
    for prompt in _PROMPTS:
        table = cache.new_request()
        logits = model.prefill(list(prompt), cache, table)
        tables.append(table)
        tokens.append(torch.argmax(logits))
    return tables, torch.stack(tokens)


def _cache(model) -> PagedKVCache:
    return PagedKVCache(
        num_layers=model.num_layers,
        num_blocks=32,
        block_size=4,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        dtype=model.dtype,
    )


def test_decode_many_uses_page_plan_when_backend_supports_it(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = load_model_runtime("esme", bundle_path=bundle).model
    backend = PageTableReferenceAttention()
    paged = load_model_runtime("esme", bundle_path=bundle, attention_backend=backend).model

    baseline_cache = _cache(baseline)
    paged_cache = _cache(paged)
    baseline_tables, tokens = _prefill(baseline, baseline_cache)
    paged_tables, _ = _prefill(paged, paged_cache)
    planned_reads = []
    plan_read_many = paged_cache.plan_read_many

    def track_read_plan(*args, **kwargs):
        read_plan = plan_read_many(*args, **kwargs)
        planned_reads.append(read_plan)
        return read_plan

    monkeypatch.setattr(paged_cache, "plan_read_many", track_read_plan)

    expected = baseline.decode_many(baseline_cache, baseline_tables, tokens)
    actual = paged.decode_many(paged_cache, paged_tables, tokens)

    torch.testing.assert_close(actual, expected)
    assert len(planned_reads) == 1
    assert planned_reads[0].idx is None
    assert planned_reads[0].cu_seqlens is None
    assert planned_reads[0].page_plan is not None
    assert backend.plan_calls == 1
    assert backend.paged_calls == paged.num_layers
    assert backend.packed_calls == 0

def test_planned_window_uses_page_plan_when_backend_supports_it(bundle: Path) -> None:
    baseline = load_model_runtime("esme", bundle_path=bundle).model
    backend = PageTableReferenceAttention()
    paged = load_model_runtime("esme", bundle_path=bundle, attention_backend=backend).model

    baseline_cache = _cache(baseline)
    paged_cache = _cache(paged)
    baseline_tables, tokens = _prefill(baseline, baseline_cache)
    paged_tables, _ = _prefill(paged, paged_cache)
    baseline_plan = baseline.open_decode_window(baseline_cache, baseline_tables, budget=2)
    paged_plan = paged.open_decode_window(paged_cache, paged_tables, budget=2)
    assert baseline_plan is not None
    assert paged_plan is not None

    expected = baseline.decode_window_step(baseline_cache, baseline_plan, tokens)
    actual = paged.decode_window_step(paged_cache, paged_plan, tokens)

    torch.testing.assert_close(actual, expected)
    assert backend.plan_calls == 1
    assert backend.paged_calls == paged.num_layers
    assert backend.packed_calls == 0
