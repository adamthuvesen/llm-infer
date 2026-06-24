"""Request queue, per-request token sampling, and the continuous-batching decode loop.

Phase B filled the minimal vertical-slice runner here (engine + request + greedy
selection). Sampling is **per request**: each :class:`Request` carries its own
:class:`SamplingParams` (temperature, top-p, top-k, presence/frequency penalties, seed) and
draws from its own seeded generator, so a sampled request reproduces its tokens identically
run alone or batched. Greedy stays the default and the proven oracle path.
"""

from __future__ import annotations

from llm_infer.serving.engine import InferenceEngine, StepResult
from llm_infer.serving.request import Request
from llm_infer.serving.sampler import GREEDY, SamplingParams, greedy, sample_row
from llm_infer.serving.speculative import PromptLookupDraft, SpeculativeDecodingConfig

__all__ = [
    "GREEDY",
    "InferenceEngine",
    "PromptLookupDraft",
    "Request",
    "SamplingParams",
    "SpeculativeDecodingConfig",
    "StepResult",
    "greedy",
    "sample_row",
]
