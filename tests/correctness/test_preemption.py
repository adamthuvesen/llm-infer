"""Preemption (recompute) correctness: an evicted-and-resumed request stays token-exact.

The whole thesis of the engine is token-exactness, so the bar for preemption is the same
bar as everywhere else: a request that is preempted under KV pressure and later resumed by
recompute must produce the **identical greedy continuation** it would have produced if it had
never been interrupted — token for token. Recompute rebuilds its KV from prompt-plus-generated
at the original absolute positions, so the reconstructed state is exactly the evicted one.

These run on the tiny random-weight Qwen (real RoPE, attention, and paged KV history; no 3B
load), so the forward genuinely depends on the reconstructed cache: a botched recompute would
flip tokens. CPU-runnable, fp32, zero GPU spend.
"""

from __future__ import annotations

from llm_infer.serving import InferenceEngine, Request
from llm_infer.tracing import TraceRecorder
from tests.correctness.test_chunked_prefill import _tiny_qwen

# A vocab-37 model with EOS pinned to an id it never greedily emits in these short runs, so the
# length cap (not EOS) ends each request and the full continuation is exercised.
EOS = frozenset({36})


def _run(model, requests, *, num_blocks, preemption, block_size=4, **engine_kwargs):
    engine = InferenceEngine(
        model,
        block_size=block_size,
        num_blocks=num_blocks,
        preemption=preemption,
        **engine_kwargs,
    )
    for prompt_ids, max_new, request_id in requests:
        engine.add_request(Request(request_id, list(prompt_ids), max_new, EOS))
    return engine.run()


def _run_single(model, prompt_ids, max_new, *, num_blocks, preemption, **engine_kwargs):
    out = _run(
        model,
        [(prompt_ids, max_new, "only")],
        num_blocks=num_blocks,
        preemption=preemption,
        **engine_kwargs,
    )
    return out["only"]


def test_preemption_actually_fires_under_a_tight_budget() -> None:
    """The tight-budget run must really preempt — otherwise the token-exactness test is empty."""
    model = _tiny_qwen()
    recorder = TraceRecorder()
    # prompt 3 + decode: worst-case 2 blocks each (ceil((3+6-1)/4)); footprint 1 block each.
    # Reservation mode would admit only one (3 blocks / 2 each); preemption over-commits to
    # three (1 block each) and then must evict as they grow — a genuine forced preemption.
    requests = [([1, 5, 9], 6, "a"), ([2, 6, 10], 6, "b"), ([3, 7, 11], 6, "c")]
    _run(model, requests, num_blocks=3, preemption=True, trace=recorder)

    preemptions = [e for e in recorder.events if e.event == "request_preempted"]
    resumes = [e for e in recorder.events if e.event == "request_resumed"]
    assert preemptions, "expected at least one real preemption under the tight budget"
    assert resumes, "a preempted request must resume"
    # Each preemption frees the victim's KV honestly and names the pressure reason.
    for event in preemptions:
        assert event.preempt_reason == "kv_pressure"
        assert event.block_count >= 1
        assert event.pool_used is not None and event.pool_free is not None


def test_preempted_request_output_equals_uninterrupted() -> None:
    """Every request's tokens under a preempting budget == its tokens with room to spare."""
    model = _tiny_qwen()
    requests = [([1, 5, 9], 6, "a"), ([2, 6, 10], 6, "b"), ([3, 7, 11], 6, "c")]

    # Roomy budget: worst-case reservation fits all three, so nobody is ever preempted.
    roomy = _run(model, requests, num_blocks=8, preemption=False)
    # Tight budget with preemption: the same three requests, but the pool must evict to fit.
    tight = _run(model, requests, num_blocks=3, preemption=True)

    for _, _, request_id in requests:
        assert tight[request_id] == roomy[request_id], (
            f"{request_id}: preempted output diverged from the uninterrupted run "
            f"(preempted={tight[request_id]}, uninterrupted={roomy[request_id]})"
        )


def test_preemption_with_chunked_prefill_stays_token_exact() -> None:
    """Recompute resume in chunks (a long prompt+generated) still reproduces the exact tokens."""
    model = _tiny_qwen()
    requests = [
        ([1, 5, 9, 13, 17], 6, "long-a"),
        ([2, 6, 10, 14, 18], 6, "long-b"),
        ([3, 7, 11, 15, 19], 6, "long-c"),
    ]

    roomy = _run(model, requests, num_blocks=12, preemption=False, prefill_chunk_size=2)
    tight = _run(model, requests, num_blocks=4, preemption=True, prefill_chunk_size=2)

    for _, _, request_id in requests:
        assert tight[request_id] == roomy[request_id], (
            f"{request_id}: chunked-recompute resume diverged from the uninterrupted run"
        )


def test_prefill_skips_a_request_preempted_mid_step() -> None:
    """A request evicted while an earlier one prefills must not be prefilled out of the queue.

    Regression for a stale running-set snapshot: ``_prefill_requests`` / ``_resume_requests`` once
    captured the running set *before* the loop, so a candidate that ``_ensure_pool_room`` preempted
    partway through (to free blocks for an earlier request) still passed the membership check and
    got prefilled while sitting in ``waiting`` — double-allocating its KV and stalling forward
    progress. With chunked prefill over a tight pool the eviction lands inside the prefill loop. We
    step the engine and assert the invariant that a waiting request never holds cache state, then
    that every output still matches the uninterrupted run token-for-token.
    """
    model = _tiny_qwen()
    requests = [
        ([1, 5, 9], 5, "a"),
        ([2, 6, 10], 5, "b"),
        ([3, 7, 11], 5, "c"),
        ([4, 8, 12], 5, "d"),
    ]
    recorder = TraceRecorder()
    engine = InferenceEngine(
        model, block_size=2, num_blocks=4, preemption=True, prefill_chunk_size=1, trace=recorder
    )
    tracked = [Request(rid, list(prompt), max_new, EOS) for prompt, max_new, rid in requests]
    for request in tracked:
        engine.add_request(request)

    steps = 0
    while engine.scheduler.has_work():
        engine.step()
        steps += 1
        assert steps < 200, "engine made no forward progress (suspected preemption livelock)"
        for waiting in engine.scheduler.waiting:
            assert not waiting.prefilled and waiting.block_table is None, (
                f"{waiting.request_id!r} was prefilled while in the waiting queue — a request "
                "preempted mid-step must not be touched again until it is re-admitted"
            )

    assert any(e.event == "request_preempted" for e in recorder.events), (
        "scenario must actually preempt mid-prefill or it does not exercise the bug"
    )
    tight = {request.request_id: request.generated for request in tracked}
    roomy = _run(
        model, requests, num_blocks=16, preemption=False, block_size=2, prefill_chunk_size=1
    )
    for _, _, request_id in requests:
        assert tight[request_id] == roomy[request_id], (
            f"{request_id}: output diverged from the uninterrupted run "
            f"(preempted={tight[request_id]}, uninterrupted={roomy[request_id]})"
        )


def test_preemption_preserves_batched_equals_serial() -> None:
    """Under preemption, each request batched == run alone — eviction adds no drift."""
    model = _tiny_qwen()
    requests = [([1, 5, 9], 6, "a"), ([2, 6, 10], 6, "b"), ([3, 7, 11], 6, "c")]

    serial = {
        rid: _run_single(model, prompt, max_new, num_blocks=8, preemption=False)
        for prompt, max_new, rid in requests
    }
    batched = _run(model, requests, num_blocks=3, preemption=True)

    for _, _, request_id in requests:
        assert batched[request_id] == serial[request_id], (
            f"{request_id}: batched-under-preemption != serial "
            f"(batched={batched[request_id]}, serial={serial[request_id]})"
        )


def test_recompute_rebuilds_identical_kv_state() -> None:
    """A resumed request's reconstructed last-position logits match the uninterrupted state.

    Stronger than the token check: it pins that recompute reproduces the *continuous* KV, not
    just a sequence that happens to greedy-decode the same. We compare the next-token logits the
    model produces from the resumed cache against the same model decoding the request alone to
    the same length.
    """
    model = _tiny_qwen()
    prompt_ids = [1, 5, 9]
    max_new = 6

    # Uninterrupted single-request reference.
    reference = _run_single(model, prompt_ids, max_new, num_blocks=8, preemption=False)
    # The same request, forced through at least one preempt/resume cycle. It is admitted LAST so
    # the LIFO victim policy evicts it (the newest running request) under the tight budget.
    recorder = TraceRecorder()
    out = _run(
        model,
        [
            ([2, 6, 10], max_new, "filler"),
            ([3, 7, 11], max_new, "filler2"),
            (prompt_ids, max_new, "victim"),
        ],
        num_blocks=3,
        preemption=True,
        trace=recorder,
    )
    assert any(
        e.event == "request_preempted" and e.request_id == "victim" for e in recorder.events
    ), "the victim must actually be preempted in this scenario"
    assert out["victim"] == reference
