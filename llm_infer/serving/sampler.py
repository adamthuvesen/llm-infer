"""Token selection from next-token logits: greedy and seeded temperature/top-p sampling.

Phase B shipped greedy only. Phase E adds temperature + nucleus (top-p) multinomial
sampling for the rlvr-sql GRPO rollout workload, which decodes at ``temperature=1.0``.
The sampler holds a seeded :class:`torch.Generator`, so a rollout is reproducible given the
seed and the (fixed) admission/decode schedule.

``temperature == 0`` is a hard special case that returns the argmax — token-for-token
identical to :func:`greedy`, the path the Phase A/B greedy oracle already proves — so routing
the engine through the sampler leaves the greedy correctness suite unchanged. The temperature
scaling and softmax run in fp32 regardless of the model dtype, so sampling is numerically
stable under the bf16 weights it serves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def greedy(logits: torch.Tensor) -> int:
    """The argmax token id from a 1-D ``(vocab_size,)`` logit row."""
    if logits.ndim != 1:
        raise ValueError(f"expected 1-D logits, got shape {tuple(logits.shape)}")
    return int(torch.argmax(logits).item())


@dataclass
class Sampler:
    """Greedy or seeded temperature/top-p multinomial token selection.

    ``temperature == 0`` → argmax (greedy), bit-for-bit :func:`greedy`. ``temperature > 0``
    scales the logits, applies nucleus (top-p) truncation, and draws from the resulting
    distribution with a seeded generator. ``top_p == 1.0`` is a no-op nucleus (the full
    softmax), supported explicitly. The generator is created lazily on the logits' device so
    a CPU unit test and a CUDA rollout share one code path.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    _generator: torch.Generator | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0; got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1]; got {self.top_p}")

    @property
    def is_greedy(self) -> bool:
        """Whether this sampler decodes greedily (temperature 0) — the proven oracle path."""
        return self.temperature == 0.0

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """One scalar token tensor from a 1-D ``(vocab,)`` row (the prefill path)."""
        if logits.ndim != 1:
            raise ValueError(f"expected 1-D logits, got shape {tuple(logits.shape)}")
        if self.is_greedy:
            return torch.argmax(logits)
        probs = self._nucleus_probs(logits.unsqueeze(0))  # (1, vocab)
        token = torch.multinomial(probs, num_samples=1, generator=self._gen(logits.device))
        return token.squeeze()

    def sample_many(self, logits: torch.Tensor) -> torch.Tensor:
        """One token tensor per row of a 2-D ``(B, vocab)`` batch, kept on device."""
        if logits.ndim != 2:
            raise ValueError(f"expected 2-D logits, got shape {tuple(logits.shape)}")
        if self.is_greedy:
            return torch.argmax(logits, dim=-1)
        probs = self._nucleus_probs(logits)  # (B, vocab)
        tokens = torch.multinomial(probs, num_samples=1, generator=self._gen(logits.device))
        return tokens.squeeze(-1)

    def _gen(self, device: torch.device) -> torch.Generator:
        """The seeded generator, created once on first use (on the logits' device).

        The device is fixed for an engine instance; a later device change would be a real bug
        (CPU/CUDA mix), so it raises rather than silently re-seeding mid-rollout.
        """
        if self._generator is None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)
        elif self._generator.device != torch.device(device):
            raise ValueError(
                f"sampler generator is on {self._generator.device}, got logits on {device}"
            )
        return self._generator

    def _nucleus_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled fp32 softmax, optional top-p nucleus truncation. ``(B, vocab)``."""
        probs = torch.softmax(logits.float() / self.temperature, dim=-1)
        if self.top_p >= 1.0:
            return probs
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep the smallest prefix whose cumulative mass reaches top_p: a token is dropped only
        # if the mass *strictly before* it already covers top_p, so the boundary token stays.
        drop = (cumulative - sorted_probs) > self.top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        kept = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        return kept / kept.sum(dim=-1, keepdim=True)
