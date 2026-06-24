"""Run the OpenAI-compatible server around the pinned Qwen model: ``python -m llm_infer.serve``.

The heavy load — 3B weights and the tokenizer — lives inside :func:`build_qwen_app` and
:func:`main`, never at import time, so importing this module (or the app package) costs
nothing. Tests build the app around the tiny CPU model directly and never call this.
"""

from __future__ import annotations

import argparse

import torch

from llm_infer.model.config import MODEL_ID, MODEL_REVISION
from llm_infer.model.qwen import QwenModel
from llm_infer.serving.engine import InferenceEngine
from llm_infer.serving.sampler import Sampler
from llm_infer.serving.server import AsyncInferenceEngine, ServerMetrics, create_app

# A 64-token page and a few hundred pages comfortably hold a handful of concurrent
# chat sessions of the lengths this server is demoed at; tune via the CLI for a real load.
DEFAULT_BLOCK_SIZE = 64
DEFAULT_NUM_BLOCKS = 512


def build_qwen_app(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
):
    """Load the pinned Qwen + tokenizer and wire the serving app around them."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = QwenModel.load(dtype=dtype, device=device)
    eos_token_ids = _eos_token_ids(tokenizer)

    sampler = Sampler()  # greedy — the proven path; swap to a seeded sampler for sampled serving
    engine = InferenceEngine(
        model, block_size=block_size, num_blocks=num_blocks, device=device, sampler=sampler
    )
    metrics = ServerMetrics()
    async_engine = AsyncInferenceEngine(engine, metrics=metrics)
    return create_app(
        async_engine=async_engine,
        tokenizer=tokenizer,
        model_id=MODEL_ID,
        eos_token_ids=eos_token_ids,
        sampler=sampler,
        metrics=metrics,
    )


def _eos_token_ids(tokenizer: object) -> frozenset[int]:
    """Generation-stopping ids: the model's generation-config eos plus the tokenizer eos."""
    from transformers import GenerationConfig

    config = GenerationConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    ids: set[int] = set()
    cfg_eos = config.eos_token_id
    if isinstance(cfg_eos, int):
        ids.add(cfg_eos)
    elif cfg_eos is not None:
        ids.update(cfg_eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return frozenset(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Qwen2.5-Coder over an OpenAI HTTP API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    args = parser.parse_args()

    import uvicorn

    app = build_qwen_app(device=args.device, block_size=args.block_size, num_blocks=args.num_blocks)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
