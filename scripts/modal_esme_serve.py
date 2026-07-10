"""Serve the Esme chat UI and OpenAI-compatible API from a Modal A100.

The same HTTP app as ``python -m llm_infer.serve``, launched on GPU instead of the local
CPU — which backend the chat UI talks to is purely a matter of how you start it:

    uv run python -m llm_infer.serve --bundle <path>     # local CPU, fp32 reference path
    modal serve scripts/modal_esme_serve.py              # A100, bf16 FlashInfer + graphs

``modal serve`` prints a temporary ``*.modal.run`` URL (hot-reloads while running;
Ctrl-C tears it down). ``modal deploy`` keeps a persistent URL instead. Either way the
UI, the ``/v1`` API, and ``/metrics`` are identical to the local server.

The bundle is read from the shared ``llm-infer-esme-bundles`` volume, which every Modal
harness in this repo stages on run. If the volume is empty, stage it once:

    modal run scripts/modal_esme_decode_profile.py --command serve-smoke

Cold start pays model load, FlashInfer warmup, and decode-graph capture (piecewise plus
the grouped serving default) — roughly a minute on A100 with the default buckets. The
container scales to zero after ``SCALEDOWN_WINDOW_S`` idle seconds, so an unused deploy
costs nothing.
"""

from __future__ import annotations

import modal

from scripts.modal_esme_bundle import ESME_BUNDLE_MOUNT, REMOTE_BUNDLE_PATH, VOLUME_NAME
from scripts.modal_flash_image import FLASH_IMAGE

# One engine serves every request through continuous batching, so one container with many
# concurrent inputs is the right shape; a second container would mean a second model copy
# and a cold KV cache, not more throughput.
MAX_CONCURRENT_REQUESTS = 64
SCALEDOWN_WINDOW_S = 5 * 60

app = modal.App("llm-infer-esme-serve")

esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
    scaledown_window=SCALEDOWN_WINDOW_S,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_REQUESTS)
@modal.asgi_app()
def serve():
    from pathlib import Path

    import torch

    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serve import build_app_from_runtime

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
    )
    return build_app_from_runtime(runtime, device="cuda")
