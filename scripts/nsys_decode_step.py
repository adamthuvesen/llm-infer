"""Child process for the decode timeline trace: one arm's fixed-length decode under nsys.

``scripts/modal_esme_decode_profile.py --command timeline`` runs this script under
``nsys profile --capture-range=cudaProfilerApi``, once per arm. Only the measured decode sits
between ``cudaProfilerStart`` and ``cudaProfilerStop``, so model load, compilation, graph capture,
warmup, and prefill stay out of the trace. ``compiled`` swaps ``rms_norm`` and ``apply_rope`` for
``torch.compile`` versions the same way the fusion probe does. Diagnostic only: no reference gate
and no speed claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import llm_infer.model.decode_graph as decode_graph_module
import llm_infer.model.pretrain_bundle as bundle_module
from llm_infer.benchmarks.esme_paged import build_requests
from llm_infer.model import rope_utils
from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serving import InferenceEngine, Request

BLOCK_SIZE = 128
WARMUP_RUNS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--arm", choices=("eager", "compiled"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    args = parser.parse_args()

    if args.arm == "compiled":
        compiled_rms_norm = torch.compile(rope_utils.rms_norm, dynamic=True)
        bundle_module.rms_norm = compiled_rms_norm
        bundle_module.apply_rope = torch.compile(rope_utils.apply_rope, dynamic=True)
        decode_graph_module.rms_norm = compiled_rms_norm

    runtime = load_model_runtime(
        "esme", bundle_path=args.bundle, dtype=torch.bfloat16, device="cuda"
    )
    requests = build_requests(runtime.tokenizer, args.batch)
    engine = InferenceEngine(
        runtime.model,
        block_size=BLOCK_SIZE,
        num_blocks=args.batch * (args.tokens // BLOCK_SIZE + 2) + 4,
        device="cuda",
        capabilities=runtime.capabilities,
    )

    def prefill() -> None:
        for req in requests:
            engine.add_request(
                Request(req.request_id, list(req.prompt_ids), args.tokens, frozenset())
            )
        engine.step()
        torch.cuda.synchronize()

    def decode() -> None:
        while engine.scheduler.has_work():
            engine.step()
        torch.cuda.synchronize()

    if args.arm == "compiled":
        # Compile the decode shapes on the eager path before capture.
        prefill()
        decode()
    enable_decode_graphs_if_cuda(runtime.model, (args.batch,))
    for _ in range(WARMUP_RUNS):
        prefill()
        decode()

    prefill()
    torch.cuda.profiler.start()
    decode()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
