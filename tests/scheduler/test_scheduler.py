"""Unit tests for continuous-batching admission — no model, pure scheduling logic.

Pins the v1 policy: admit within a block budget, head-of-line blocking, and free the
budget on finish so a queued request can take a finished one's place (the admit-after-
finish edge of the vertical slice).
"""

from __future__ import annotations

import pytest

from llm_infer.scheduler import Scheduler, max_blocks_for
from llm_infer.serving.request import Request


def _request(request_id: str, prompt_len: int = 2, max_new_tokens: int = 2) -> Request:
    return Request(
        request_id=request_id,
        prompt_ids=list(range(prompt_len)),
        max_new_tokens=max_new_tokens,
        eos_token_ids=frozenset(),
    )


def test_max_blocks_for_counts_prompt_plus_decode() -> None:
    # prompt 2 + (max_new_tokens 2 - 1) = 3 positions -> ceil(3/4) = 1 block.
    assert max_blocks_for(_request("r", prompt_len=2, max_new_tokens=2), block_size=4) == 1
    # 3 + (6 - 1) = 8 positions -> ceil(8/4) = 2 blocks.
    assert max_blocks_for(_request("r", prompt_len=3, max_new_tokens=6), block_size=4) == 2


def test_admit_respects_block_budget() -> None:
    sched = Scheduler(num_blocks=2, block_size=4)  # each request needs 1 block here
    for rid in ("a", "b", "c"):
        sched.add(_request(rid))
    admitted = [r.request_id for r in sched.admit()]
    assert admitted == ["a", "b"]  # budget fits two; "c" is held back
    assert [r.request_id for r in sched.running] == ["a", "b"]
    assert [r.request_id for r in sched.waiting] == ["c"]


def test_release_frees_budget_for_the_next_request() -> None:
    sched = Scheduler(num_blocks=2, block_size=4)
    reqs = {rid: _request(rid) for rid in ("a", "b", "c")}
    for req in reqs.values():
        sched.add(req)
    sched.admit()  # a, b running
    assert not sched.admit()  # c still blocked, budget full

    sched.release(reqs["a"])
    admitted = [r.request_id for r in sched.admit()]
    assert admitted == ["c"]  # a's freed budget lets c in


def test_admission_is_head_of_line() -> None:
    """A big request at the front blocks smaller ones behind it (FIFO, no reordering)."""
    sched = Scheduler(num_blocks=2, block_size=4)
    big = _request("big", prompt_len=3, max_new_tokens=3)  # 3+3-1=5 positions -> 2 blocks
    assert max_blocks_for(big, block_size=4) == 2
    small = _request("small")  # needs 1 block
    sched.add(_request("filler"))  # 1 block, admitted first
    sched.add(big)
    sched.add(small)
    admitted = [r.request_id for r in sched.admit()]
    assert admitted == ["filler"]  # filler takes 1 of 2 blocks; big needs 2 -> blocked; stop
    assert [r.request_id for r in sched.waiting] == ["big", "small"]


def test_add_rejects_request_too_large_for_pool() -> None:
    sched = Scheduler(num_blocks=1, block_size=4)
    too_big = _request("x", prompt_len=4, max_new_tokens=6)  # needs 2 blocks > pool of 1
    with pytest.raises(ValueError, match="never be admitted"):
        sched.add(too_big)


def test_has_work_tracks_queue_and_running() -> None:
    sched = Scheduler(num_blocks=2, block_size=4)
    assert not sched.has_work()
    req = _request("a")
    sched.add(req)
    assert sched.has_work()
    sched.admit()
    sched.release(req)
    assert not sched.has_work()
