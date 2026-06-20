"""llm-infer: a minimal, honest paged LLM inference engine for Qwen2.5-Coder-3B-Instruct.

Phase A scope: the correctness oracle and the repo skeleton only — a reference
`torch_naive` attention backend and a single-request greedy decode path validated
token-for-token against HuggingFace. Paging, batching, fast kernels, and benchmarks
are later phases (see docs/scoping.md).
"""

__version__ = "0.1.0"
