"""Per-request token selection from next-token logits: greedy and seeded sampling.

Each request owns a seeded :class:`torch.Generator`, so batchmates cannot consume its random
draws. Exact tokens can still change with the decode batch shape because bf16 model kernels can
produce slightly different logits. Given the same logits and history, the draw depends only on
the request's :class:`SamplingParams` and generator state.

``temperature == 0`` is a hard special case that returns the argmax — token-for-token
the path already checked against the cached greedy reference — so a greedy request
stays exactly
the proven path, with no RNG drawn. Penalties, temperature
scaling, and the softmax run in fp32 regardless of model dtype, so sampling stays
numerically stable under the bf16 weights it serves. The per-row order matches the
standard decode surface: penalties → temperature → top-k → top-p → softmax → sample.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from llm_infer.request.sampling import SamplingParams


def sample_row(
    logits: torch.Tensor,
    params: SamplingParams,
    generated: list[int],
    generator: torch.Generator,
) -> torch.Tensor:
    """Select one token from a 1-D ``(vocab,)`` logit row under ``params``.

    Greedy rows return the argmax with no RNG. Otherwise the standard decode surface runs in
    fp32: penalties (against this request's ``generated`` history) → temperature → top-k →
    top-p → softmax → multinomial draw from ``generator``. The returned token is a scalar
    long tensor on the logits' device.
    """
    if logits.ndim != 1:
        raise ValueError(f"expected 1-D logits, got shape {tuple(logits.shape)}")
    if params.is_greedy:
        return torch.argmax(logits)

    scores = logits.float()
    scores = _apply_penalties(scores, params, generated)
    scores = scores / params.temperature
    scores = _apply_top_k(scores, params.top_k)
    probs = torch.softmax(scores, dim=-1)
    probs = _apply_top_p(probs, params.top_p)
    token = torch.multinomial(probs, num_samples=1, generator=generator)
    return token.squeeze()


def sample_rows(
    logits: torch.Tensor,
    params: Sequence[SamplingParams],
    generators: Sequence[torch.Generator | None],
    histories: Sequence[Sequence[int] | None],
) -> torch.Tensor:
    """Select one token per row of ``(B, vocab)`` logits, each row under its own request.

    The filtering surface (penalties → temperature → top-k → top-p) runs **batched** over the
    sampled rows, so its kernel count is constant in batch size. Each disabled stage is an
    exact no-op on its row, matching :func:`sample_row`. The final draw stays one
    ``torch.multinomial`` per row from that row's own generator — the per-request RNG stream
    (seed, draw count, consumption order) is identical to sampling the rows one at a time, so
    batchmates cannot consume another request's draws. Greedy rows take a vectorized argmax and
    touch no RNG. ``histories`` is read only for rows with a non-zero penalty; pass ``None``
    for the rest so callers never materialize token history just to sample.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected 2-D logits, got shape {tuple(logits.shape)}")
    rows = logits.shape[0]
    if not (rows == len(params) == len(generators) == len(histories)):
        raise ValueError(
            f"row count mismatch: {rows} logit rows, {len(params)} params, "
            f"{len(generators)} generators, {len(histories)} histories"
        )
    if all(p.is_greedy for p in params):
        return torch.argmax(logits, dim=-1)

    tokens = torch.empty(rows, dtype=torch.long, device=logits.device)
    greedy_rows = [i for i, p in enumerate(params) if p.is_greedy]
    sampled_rows = [i for i, p in enumerate(params) if not p.is_greedy]
    if greedy_rows:
        index = torch.tensor(greedy_rows, device=logits.device)
        tokens[index] = torch.argmax(logits.index_select(0, index), dim=-1)

    sampled_index = torch.tensor(sampled_rows, device=logits.device)
    scores = logits.index_select(0, sampled_index).float()
    for position, row in enumerate(sampled_rows):
        p = params[row]
        if p.presence_penalty == 0.0 and p.frequency_penalty == 0.0:
            continue
        history = histories[row]
        if history is None:
            raise ValueError(f"row {row} has penalties but no generated history was passed")
        scores[position] = _apply_penalties(scores[position], p, list(history))

    temperatures = torch.tensor(
        [params[row].temperature for row in sampled_rows],
        dtype=scores.dtype,
        device=scores.device,
    )
    scores = scores / temperatures.unsqueeze(-1)
    scores = _apply_top_k_rows(scores, [params[row].top_k for row in sampled_rows])
    probs = torch.softmax(scores, dim=-1)
    probs = _apply_top_p_rows(probs, [params[row].top_p for row in sampled_rows])

    for position, row in enumerate(sampled_rows):
        generator = generators[row]
        if generator is None:
            raise ValueError(f"row {row} is sampled but has no generator")
        tokens[row] = torch.multinomial(probs[position], num_samples=1, generator=generator)
    return tokens


def _apply_top_k_rows(scores: torch.Tensor, top_ks: list[int]) -> torch.Tensor:
    """Row-wise :func:`_apply_top_k` over a ``(B, vocab)`` batch; disabled rows are no-ops.

    A row's threshold is its ``top_k``-th highest score (strict ``<`` masking keeps ties at
    the threshold, exactly like the per-row path); a disabled row gets ``-inf``, which masks
    nothing.
    """
    vocab = scores.shape[-1]
    if not any(0 < k < vocab for k in top_ks):
        return scores
    sorted_scores = torch.sort(scores, descending=True, dim=-1).values
    kth_positions = torch.tensor(
        [k - 1 if 0 < k < vocab else vocab - 1 for k in top_ks],
        dtype=torch.long,
        device=scores.device,
    )
    kth_values = sorted_scores.gather(-1, kth_positions.unsqueeze(-1)).squeeze(-1)
    enabled = torch.tensor([0 < k < vocab for k in top_ks], dtype=torch.bool, device=scores.device)
    thresholds = torch.where(enabled, kth_values, torch.full_like(kth_values, float("-inf")))
    return scores.masked_fill(scores < thresholds.unsqueeze(-1), float("-inf"))


def _apply_top_p_rows(probs: torch.Tensor, top_ps: list[float]) -> torch.Tensor:
    """Row-wise :func:`_apply_top_p` over a ``(B, vocab)`` batch; ``top_p >= 1`` rows are no-ops.

    The nucleus rule matches the per-row path bit for bit (drop once the mass *before* a token
    already covers ``top_p``), and rows with ``top_p >= 1`` return their input probabilities
    untouched — no renormalization ever lands on a disabled row.
    """
    if all(p >= 1.0 for p in top_ps):
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    thresholds = torch.tensor(top_ps, dtype=probs.dtype, device=probs.device).unsqueeze(-1)
    drop = (cumulative - sorted_probs) >= thresholds
    kept_sorted = sorted_probs.masked_fill(drop, 0.0)
    kept = torch.zeros_like(probs).scatter_(-1, sorted_idx, kept_sorted)
    renormalized = kept / kept.sum(dim=-1, keepdim=True)
    enabled = torch.tensor(
        [p < 1.0 for p in top_ps], dtype=torch.bool, device=probs.device
    ).unsqueeze(-1)
    return torch.where(enabled, renormalized, probs)


def _apply_penalties(
    scores: torch.Tensor, params: SamplingParams, generated: list[int]
) -> torch.Tensor:
    """Subtract OpenAI-style presence/frequency penalties using this request's own history."""
    if not generated or (params.presence_penalty == 0.0 and params.frequency_penalty == 0.0):
        return scores
    counts = torch.bincount(
        torch.tensor(generated, dtype=torch.long, device=scores.device),
        minlength=scores.shape[-1],
    ).to(scores.dtype)
    appeared = (counts > 0).to(scores.dtype)
    return scores - params.frequency_penalty * counts - params.presence_penalty * appeared


def _apply_top_k(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask all but the ``top_k`` highest-scoring tokens to ``-inf`` (no-op when disabled)."""
    if top_k <= 0 or top_k >= scores.shape[-1]:
        return scores
    kth = torch.topk(scores, top_k, dim=-1).values[..., -1]
    return scores.masked_fill(scores < kth, float("-inf"))


def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Renormalize ``probs`` over the smallest nucleus whose mass reaches ``top_p``."""
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep the smallest prefix whose cumulative mass reaches top_p. A token is dropped once the
    # mass *before* it already covers top_p (``>=``, not ``>``): if the prefix is complete at
    # exactly top_p, the next token is redundant and must go, so the nucleus is the minimal set.
    drop = (cumulative - sorted_probs) >= top_p
    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
    kept = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    return kept / kept.sum(dim=-1, keepdim=True)
