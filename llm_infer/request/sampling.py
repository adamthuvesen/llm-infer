"""The sampling parameters a request carries: temperature, top-k, top-p, penalties, and seed."""

from __future__ import annotations

import math
from dataclasses import dataclass

_MIN_TORCH_SEED = -(2**63)
_MAX_TORCH_SEED = 2**64 - 1


@dataclass(frozen=True)
class SamplingParams:
    """One request's decode configuration. Default is greedy (temperature 0 → argmax).

    ``temperature == 0`` selects the argmax (the proven greedy path) and draws no RNG.
    ``temperature > 0`` scales the logits, then applies optional ``top_k`` and ``top_p``
    truncation and draws from the resulting distribution with the request's seeded generator.
    ``presence_penalty`` / ``frequency_penalty`` follow OpenAI semantics: presence subtracts a
    flat amount from any token that already appeared, frequency subtracts proportional to how
    many times it appeared. Ranges are validated loudly so a malformed request fails fast.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled (keep the full vocab)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        # Reject non-finite floats first: nan slips past every ``<``/``<=`` range check below
        # (``nan < 0`` is False), then poisons the softmax — a malformed request must fail loudly.
        for name, value in (
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite; got {value}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0; got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1]; got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0 (0 = disabled); got {self.top_k}")
        if not (-2.0 <= self.presence_penalty <= 2.0):
            raise ValueError(f"presence_penalty must be in [-2, 2]; got {self.presence_penalty}")
        if not (-2.0 <= self.frequency_penalty <= 2.0):
            raise ValueError(f"frequency_penalty must be in [-2, 2]; got {self.frequency_penalty}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):  # pyright: ignore[reportUnnecessaryIsInstance]  # seed arrives from request JSON
            raise ValueError(f"seed must be an integer; got {self.seed!r}")
        if not (_MIN_TORCH_SEED <= self.seed <= _MAX_TORCH_SEED):
            raise ValueError(
                f"seed must be in [{_MIN_TORCH_SEED}, {_MAX_TORCH_SEED}]; got {self.seed}"
            )

    @property
    def is_greedy(self) -> bool:
        """Whether this request decodes greedily (temperature 0) — the checked reference path."""
        return self.temperature == 0.0


# The engine's default sampling: greedy, token-for-token the checked reference path.
GREEDY = SamplingParams()
