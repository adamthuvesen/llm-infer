"""llm-infer: a minimal, honest paged LLM inference engine with model backends.

The correctness oracle comes first: a reference `torch_naive` attention backend and a
single-request greedy decode path validated token-for-token against a trusted backend
oracle. On top of it sit a paged KV-cache, continuous batching, serving, benchmarks,
and concrete Qwen / llm-pretrain DenseBackbone backends.
"""

__version__ = "0.1.0"
