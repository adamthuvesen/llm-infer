"""llm-infer: a minimal, honest paged LLM inference engine for Qwen2.5-Coder-3B-Instruct.

The correctness oracle comes first: a reference `torch_naive` attention backend and a
single-request greedy decode path validated token-for-token against HuggingFace (Phase
A). On top of it sit a paged KV-cache, a continuous-batching scheduler, and a minimal
serving loop (Phase B), plus the fused `flash_attn_paged` backend gated by the same
oracle (Phase C). Benchmarks and the rlvr-sql rollout hook are later phases (see
docs/scoping.md).
"""

__version__ = "0.1.0"
