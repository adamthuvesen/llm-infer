"""llm-infer: a minimal, small paged LLM inference engine with model backends.

The reference check comes first: a reference `torch_naive` attention backend and a
single-request greedy decode path validated token-for-token against a trusted backend
reference check. On top of it sit a paged KV-cache, continuous batching, serving, benchmarks,
and concrete Qwen / Esme backends.
"""

__version__ = "0.1.0"
