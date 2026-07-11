"""Modal A100 harness: validate the Esme flash-attn KERNEL at equal dtype (bf16 vs bf16).

Esme had only ever been validated on the ``torch_naive`` backend; ``FlashAttnPagedAttention`` is
GPU-only and was never exercised on Esme weights (``tests/correctness/test_flash_attn_paged.py`` is
Qwen-only). To validate flash for Esme, compare the **flash kernel against torch_naive at the same
dtype** — both bf16, through the same engine paged path. That isolates the kernel: any difference
is the flash kernel alone, not dtype.

The corrected contract compares bf16 flash against bf16 ``torch_naive``. An earlier version used
the **fp32** oracle, so kernel behavior and whole-model bf16 rounding were mixed. Genuine near-ties
were flipped on Esme's thin-margin QK-norm logits (esme-001 step 22: fp32 gap 0.0119), which made
that comparison unsuitable for checking the kernel. The recorded step-22 investigation found
exact agreement between bf16 flash and bf16 ``torch_naive`` (gap 0.0000). Exact token parity is
therefore required at equal dtype. A divergence is treated as a flash-kernel bug.

This mirrors the Qwen flash gate (``tests/correctness/test_flash_attn_paged.py``), which compares
bf16 flash to a bf16 golden, and ``scripts/modal_esme_reference_check.py`` (the Esme torch_naive
gate, same bundle staging). The flash image is the shared ``scripts/modal_flash_image.py``
definition every flash harness imports, so flash-attn installs from a prebuilt wheel once.

    modal run scripts/modal_esme_flash_reference_check.py --command smoke
    modal run scripts/modal_esme_flash_reference_check.py --command check
"""

from __future__ import annotations

from pathlib import Path

import modal

from scripts.modal_esme_bundle import (
    ESME_BUNDLE_MOUNT,
    REMOTE_BUNDLE_PATH,
    VOLUME_NAME,
    local_bundle_path,
    stage_bundle,
)
from scripts.modal_flash_image import FLASH_IMAGE

BLOCK_SIZE = 128

app = modal.App("llm-infer-esme-flash-reference-check")

# One shared flash-attn image for every GPU harness (Qwen + Esme). flash-attn installs from a
# prebuilt wheel (no source compile, no OOM) — see scripts/modal_flash_image.py.
flash_image = FLASH_IMAGE

esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=flash_image,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=30 * 60,
)
def check_esme_flash(num_requests: int, max_new_tokens: int) -> str:
    """Fail if the Esme flash kernel diverges from bf16 torch_naive at equal dtype (exact ids).

    Loads Esme twice in **bf16**, both through the same engine paged path: once on
    ``FlashAttnPagedAttention`` (the kernel under test) and once on the default ``torch_naive``
    (the equal-dtype reference). The kernel is isolated by using the same dtype on both sides.
    Exact token parity is required. The recorded step-22 investigation found a gap of 0.0000
    between bf16 flash and bf16 ``torch_naive``. A mismatch is treated as an Esme flash-kernel
    bug and must be reported.
    """
    import torch

    from llm_infer.benchmarks import normalize_at_eos
    from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serving import InferenceEngine, Request

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"

    flash_runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
        attention_backend=FlashAttnPagedAttention(),
    )
    if not flash_runtime.capabilities.flash_attention:
        raise AssertionError("Esme flash runtime did not advertise flash_attention")
    if flash_runtime.eos_token_ids != frozenset({2}):
        raise AssertionError(f"expected Esme EOS [2], got {sorted(flash_runtime.eos_token_ids)}")

    # The equal-dtype reference: bf16 on the default torch_naive backend, same engine path.
    naive_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.bfloat16, device="cuda"
    )

    prompts = (
        "Write a tiny Python function that doubles an integer.",
        "Explain KV caching in one short sentence.",
        "Give one SQL query that counts rows in a table named events.",
        "Name two practical checks before trusting a benchmark.",
    )
    eos = flash_runtime.eos_token_ids
    requests: list[tuple[str, list[int]]] = []
    for index in range(num_requests):
        tokenized = flash_runtime.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[index % len(prompts)]}],
            add_generation_prompt=True,
            tokenize=True,
        )
        if not isinstance(tokenized, list) or not tokenized:
            raise ValueError(f"Esme tokenizer returned invalid prompt ids for request {index}")
        requests.append((f"esme-{index:03d}", [int(token_id) for token_id in tokenized]))

    def decode_all(runtime) -> dict[str, list[int]]:
        engine = InferenceEngine(
            runtime.model,
            block_size=BLOCK_SIZE,
            num_blocks=len(requests) + 8,
            device="cuda",
            capabilities=runtime.capabilities,
        )
        for request_id, prompt_ids in requests:
            engine.add_request(Request(request_id, list(prompt_ids), max_new_tokens, eos))
        return engine.run()

    flash_outputs = decode_all(flash_runtime)
    naive_outputs = decode_all(naive_runtime)

    exact = 0
    failures: list[str] = []
    for request_id, _ in requests:
        flash_ids = normalize_at_eos(flash_outputs[request_id], eos)
        naive_ids = normalize_at_eos(naive_outputs[request_id], eos)
        if flash_ids == naive_ids:
            exact += 1
        else:
            first = next(
                (i for i, (a, b) in enumerate(zip(flash_ids, naive_ids, strict=False)) if a != b),
                min(len(flash_ids), len(naive_ids)),
            )
            failures.append(f"{request_id} first diff at step {first}")

    if failures:
        raise AssertionError(
            "Esme flash KERNEL diverged from bf16 torch_naive at equal dtype "
            f"(real flash-kernel bug for Esme): {failures[:3]}"
        )
    return (
        f"Esme flash kernel check PASSED: {exact}/{len(requests)} exact (token-for-token) "
        "vs bf16 torch_naive at equal dtype — the flash kernel is validated for Esme"
    )


@app.local_entrypoint()
def main(command: str = "smoke", bundle_path: str = "") -> None:
    """Run the cheap smoke or the fuller Esme flash reference check on the A100.

    smoke:   modal run scripts/modal_esme_flash_reference_check.py --command smoke
    check:   modal run scripts/modal_esme_flash_reference_check.py --command check
    """
    if command == "smoke":
        num_requests, max_new_tokens = 2, 8
    elif command == "check":
        num_requests, max_new_tokens = 8, 64
    else:
        raise ValueError(f"command must be 'smoke' or 'check', got {command!r}")
    stage_bundle(esme_bundles, local_bundle_path(bundle_path), label="esme-flash")
    print(check_esme_flash.remote(num_requests, max_new_tokens))
