"""Esme three-way benchmark: naive HF vs llm_infer vs vLLM, all on Esme-214M-Chat.

The own-model mirror of the Qwen three-way run. Esme has no public HF/vLLM checkpoint, so
``scripts/convert_esme_to_hf.py`` first emits a ``Qwen3ForCausalLM`` checkpoint from the bundle
(``Esme-214M-Chat`` is architecturally Qwen3); the parity test gates that conversion against the
bundle oracle. This module then times three systems on one greedy workload:

* ``hf_sequential`` — HF ``Qwen3ForCausalLM.generate()`` once per request, sequentially. The naive
  baseline (HF's own cached generate, no cross-request batching).
* ``llm_infer`` — this engine's Esme paged-KV path on the bundle: all requests in one paged cache,
  every running request advanced in one fused batched decode per step. The attention backend is
  whatever the supplied runtime carries — on GPU the default CUDA Esme row uses FlashInfer paged
  attention, while CPU/fp32 rows fall back to the bundle's ``torch_naive`` reference path.
* ``vllm`` — vLLM offline generate on the converted checkpoint, prefix caching off. The ceiling.

Agreement uses policy v2 against the fp32 reference. Raw timing is retained, while public tok/s
requires an exact or accepted numerical status. This is why
the ``llm_infer`` row can run on a bf16 CUDA attention backend: bf16 flips like the esme-001
step-22 case (fp32 gap 0.0119, far under the 0.1 bf16 tolerance) are genuine ties, not bugs —
exactly the whole-model bf16 rounding seen in the dtype check. The pieces here are pure (no Modal,
no vLLM import at module scope) so the HF-vs-llm_infer comparison can also run on CPU when the GPU
three-way is deferred.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from llm_infer.benchmarks.esme_paged import (
    EsmeBenchRequest,
    SystemTiming,
    _time,
    reference_outputs,
)
from llm_infer.benchmarks.report import normalize_at_eos, total_output_tokens
from llm_infer.model.runtime import ModelRuntime
from llm_infer.serving import InferenceEngine, Request
from llm_infer.validation.tie_tolerance import LogitsOracle

DecodeOnce = Callable[[], dict[str, list[int]]]

# The audited bf16 automatic-acceptance boundary. Larger numerical differences require review;
# they are not automatically classified as implementation bugs. This is not the fp32 1e-3 rule.
BF16_AGREEMENT_TOLERANCE = 0.1


def run_hf_sequential_esme(
    hf_checkpoint: Path | str,
    requests: list[EsmeBenchRequest],
    *,
    max_new_tokens: int,
    eos_token_ids: frozenset[int],
    device: str,
) -> DecodeOnce:
    """Build the per-request HF greedy ``generate()`` closure for the converted Qwen3 checkpoint."""
    from transformers import AutoModelForCausalLM

    eos = sorted(eos_token_ids)
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(hf_checkpoint), dtype=_hf_dtype(device), local_files_only=True
        )
        .eval()
        .to(device)
    )

    def decode_once() -> dict[str, list[int]]:
        outputs: dict[str, list[int]] = {}
        for req in requests:
            input_ids = torch.tensor([req.prompt_ids], device=device)
            generated = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=eos,
                pad_token_id=eos[0],
            )
            outputs[req.request_id] = generated[0, input_ids.shape[1] :].tolist()
        return outputs

    return decode_once


def build_vllm_llm(
    hf_checkpoint: Path | str,
    *,
    max_model_len: int,
    gpu_memory_utilization: float = 0.6,
) -> tuple[object, dict[str, object]]:
    """Build ONE vLLM engine (prefix caching off) plus its pinned config.

    A process gets exactly one engine: vLLM's memory reservation
    (``gpu_memory_utilization``) is never returned while the process lives, so a second
    ``LLM()`` in the same process fails engine-core startup on free-memory checks.
    Multi-workload runs reuse this engine via :func:`vllm_decode_closure`.
    """
    import vllm
    from vllm import LLM

    llm = LLM(
        model=str(hf_checkpoint),
        dtype="bfloat16",
        enable_prefix_caching=False,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    config = {
        "version": vllm.__version__,
        "dtype": "bfloat16",
        "enable_prefix_caching": False,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "tensor_parallel_size": 1,
        "served_checkpoint": "converted Qwen3 (from Esme bundle)",
    }
    return llm, config


def vllm_decode_closure(
    llm: object,
    requests: list[EsmeBenchRequest],
    *,
    max_new_tokens: int,
    eos_token_ids: frozenset[int],
    ignore_eos: bool = False,
) -> DecodeOnce:
    """The greedy offline-generate closure for one workload on an already-built engine."""
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        n=1,
        stop_token_ids=[] if ignore_eos else sorted(eos_token_ids),
        ignore_eos=ignore_eos,
    )
    prompts = [{"prompt_token_ids": list(req.prompt_ids)} for req in requests]
    ids_by_index = [req.request_id for req in requests]

    def decode_once() -> dict[str, list[int]]:
        results = llm.generate(prompts, params, use_tqdm=False)  # type: ignore[attr-defined]
        return {
            ids_by_index[i]: [int(token) for token in results[i].outputs[0].token_ids]
            for i in range(len(results))
        }

    return decode_once


def run_vllm_esme(
    hf_checkpoint: Path | str,
    requests: list[EsmeBenchRequest],
    *,
    max_new_tokens: int,
    eos_token_ids: frozenset[int],
    gpu_memory_utilization: float = 0.6,
) -> tuple[DecodeOnce, dict[str, object]]:
    """Build the vLLM greedy offline-generate closure (prefix caching off) and its pinned config."""
    max_model_len = max(len(req.prompt_ids) for req in requests) + max_new_tokens
    llm, config = build_vllm_llm(
        hf_checkpoint,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    decode_once = vllm_decode_closure(
        llm, requests, max_new_tokens=max_new_tokens, eos_token_ids=eos_token_ids
    )
    return decode_once, config


def _hf_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


def _llm_infer_decode(
    runtime: ModelRuntime,
    requests: list[EsmeBenchRequest],
    *,
    max_new_tokens: int,
    block_size: int,
    num_blocks: int,
    device: str,
) -> DecodeOnce:
    eos = runtime.eos_token_ids

    def decode_once() -> dict[str, list[int]]:
        engine = InferenceEngine(
            runtime.model,
            block_size=block_size,
            num_blocks=num_blocks,
            device=device,
            capabilities=runtime.capabilities,
        )
        for req in requests:
            engine.add_request(Request(req.request_id, list(req.prompt_ids), max_new_tokens, eos))
        return engine.run()

    return decode_once


@dataclass(frozen=True)
class EsmeAgreement:
    """One system's tie-tolerant agreement profile against the fp32 oracle (the Qwen pattern)."""

    exact: int
    tie: int
    nontie: int
    total: int
    ties_sample: list[dict[str, object]]
    divergences_sample: list[dict[str, object]]
    review_required: int = 0
    failed: int = 0
    numerical_evidence: list[dict[str, object]] = field(default_factory=list)

    @property
    def all_ties_or_exact(self) -> bool:
        """True iff every divergence from the fp32 oracle is a genuine tie (or none) — agreement."""
        return self.nontie == 0


def tie_tolerant_agreement(
    oracle_model: LogitsOracle,
    requests: list[EsmeBenchRequest],
    outputs: dict[str, list[int]],
    reference: dict[str, list[int]],
    eos_token_ids: frozenset[int],
    *,
    tolerance: float = BF16_AGREEMENT_TOLERANCE,
) -> EsmeAgreement:
    """Classify a system's tokens vs the fp32 oracle for policy-v2 review.

    For each request, exact match counts as exact; otherwise the first divergence is recomputed on
    the fp32 oracle via ``compare_under_tie_tolerance``. Within ``tolerance`` it is automatically
    accepted numerical behavior; beyond it needs review unless the mismatch is structural.
    """
    from llm_infer.validation.tie_tolerance import compare_under_tie_tolerance

    prompts_by_id = {req.request_id: list(req.prompt_ids) for req in requests}
    exact = 0
    ties: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    missing = sorted(set(reference) - set(outputs))
    extra = sorted(set(outputs) - set(reference))
    if missing:
        failures.append({"request": missing[0], "detail": f"missing outputs for {missing[:3]}"})
    if extra:
        failures.append({"request": extra[0], "detail": f"unexpected outputs for {extra[:3]}"})
    for request_id in reference:
        if request_id not in outputs:
            continue
        raw = outputs[request_id]
        fast = normalize_at_eos(raw, eos_token_ids)
        gold = normalize_at_eos(reference[request_id], eos_token_ids)
        if fast == gold:
            exact += 1
            continue
        result = compare_under_tie_tolerance(
            oracle_model, prompts_by_id[request_id], fast, gold, tolerance=tolerance
        )
        if result.ok and result.divergence is not None:
            d = result.divergence
            ties.append(
                {
                    "request": request_id,
                    "step": d.step,
                    "fast_token": d.fast_token,
                    "golden_token": d.golden_token,
                    "fp32_margin": abs(d.fast_logit - d.golden_logit),
                    "reference_gap": d.reference_gap,
                    "fast_logit": d.fast_logit,
                    "golden_logit": d.golden_logit,
                }
            )
        elif not result.ok:
            numerical = compare_under_tie_tolerance(
                oracle_model,
                prompts_by_id[request_id],
                fast,
                gold,
                tolerance=float("inf"),
            )
            if numerical.divergence is None:
                failures.append({"request": request_id, "detail": result.failure})
                continue
            d = numerical.divergence
            reviews.append(
                {
                    "request": request_id,
                    "step": d.step,
                    "fast_token": d.fast_token,
                    "golden_token": d.golden_token,
                    "fp32_margin": abs(d.fast_logit - d.golden_logit),
                    "reference_gap": d.reference_gap,
                    "fast_logit": d.fast_logit,
                    "golden_logit": d.golden_logit,
                    "automatic_boundary": tolerance,
                    "detail": result.failure,
                }
            )
    divergences = [*reviews, *failures]
    return EsmeAgreement(
        exact=exact,
        tie=len(ties),
        nontie=len(divergences),
        total=len(reference),
        ties_sample=ties[:3],
        divergences_sample=divergences[:3],
        review_required=len(reviews),
        failed=len(failures),
        numerical_evidence=[*ties, *reviews],
    )


def run_three_way(
    runtime: ModelRuntime,
    hf_checkpoint: Path | str,
    requests: list[EsmeBenchRequest],
    *,
    max_new_tokens: int,
    block_size: int,
    num_blocks: int,
    warmup: int,
    iters: int,
    device: str,
    include_vllm: bool,
    llm_infer_runtime: ModelRuntime | None = None,
) -> tuple[list[SystemTiming], dict[str, EsmeAgreement]]:
    """Time naive HF, llm_infer, and (optionally) vLLM on Esme; gate each on the fp32 oracle.

    ``runtime`` is the fp32 Esme bundle runtime — it is always the **reference oracle**. The
    ``llm_infer`` row runs on ``llm_infer_runtime`` when given (for example, the bf16 CUDA fast
    runtime), else on ``runtime`` itself. ``hf_checkpoint`` is the converted Qwen3 directory the
    HF and vLLM legs load. vLLM is skipped when ``include_vllm`` is false (CPU-only /
    GPU-deferred).

    Agreement is **tie-tolerant** (the Qwen pattern): ``matches_reference`` is true when a system's
    only divergences from the fp32 oracle are genuine numerical ties. Returns one
    :class:`SystemTiming` per system plus the per-system :class:`EsmeAgreement` profile.
    """
    eos = runtime.eos_token_ids
    reference = reference_outputs(
        runtime.model, requests, max_new_tokens=max_new_tokens, eos_token_ids=eos
    )
    sync = device == "cuda"

    engine_runtime = llm_infer_runtime or runtime
    backend_label = type(engine_runtime.model.backend).__name__
    llm_infer_mode = f"Esme paged KV + batched decode ({backend_label})"

    systems: list[tuple[str, str, DecodeOnce]] = [
        (
            "hf_sequential",
            "per-request HF Qwen3 generate()",
            run_hf_sequential_esme(
                hf_checkpoint,
                requests,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos,
                device=device,
            ),
        ),
        (
            "llm_infer",
            llm_infer_mode,
            _llm_infer_decode(
                engine_runtime,
                requests,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                num_blocks=num_blocks,
                device=device,
            ),
        ),
    ]
    if include_vllm:
        vllm_decode, _ = run_vllm_esme(
            hf_checkpoint, requests, max_new_tokens=max_new_tokens, eos_token_ids=eos
        )
        systems.append(("vllm", "vLLM offline generate (ceiling)", vllm_decode))

    timings: list[SystemTiming] = []
    agreements: dict[str, EsmeAgreement] = {}
    for system, mode, decode_once in systems:
        median_s, outputs = _time(decode_once, warmup=warmup, iters=iters, sync=sync)
        agreement = tie_tolerant_agreement(runtime.model, requests, outputs, reference, eos)
        agreements[system] = agreement
        timings.append(
            SystemTiming(
                system=system,
                mode=mode,
                matches_reference=agreement.all_ties_or_exact,
                median_seconds=median_s,
                total_output_tokens=total_output_tokens(outputs, eos),
                outputs=outputs,
            )
        )
    return timings, agreements
