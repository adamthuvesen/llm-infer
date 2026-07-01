"""External-baseline parity: the converted HF Qwen3 checkpoint == the Esme bundle oracle.

``scripts/convert_esme_to_hf.py`` maps an Esme ``llm_pretrain_dense_v1`` bundle onto
``Qwen3ForCausalLM``. This is the make-or-break gate for that conversion: the converted model,
loaded by HF ``transformers`` in fp32, must reproduce ``PretrainBundleModel.logits()`` on the real
bundle — greedy token ids token-for-token, logits within documented fp32 reduction-order noise. If
HF diverges, the conversion is wrong (a bad key remap or a missed feature); the fix is the
conversion, never the tolerance.

HF CPU parity is the hard, CPU-runnable gate and runs whenever ``ESME_BUNDLE_PATH`` points at the
real ``Esme-214M-Chat`` bundle. The vLLM leg asserts the same contract — the converted checkpoint
loads under vLLM and its greedy ids match the oracle — but vLLM needs a CUDA GPU, so it is skipped
when CUDA or vLLM is unavailable (the documented GPU-deferred path).

The token-id assertion is the gate; the logit tolerance is the documented floor. On the real
weights the HF-vs-oracle logit gap is fp32 BLAS reduction-order noise (~4e-5 max measured),
hundreds of times below the model's tightest observed top-2 decision margin (2.6e-3), so it can
never flip a greedy argmax.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from llm_infer.model.decode import greedy_decode
from llm_infer.model.pretrain_bundle import PretrainBundleModel
from llm_infer.model.runtime import load_model_runtime

# fp32 BLAS reduction-order noise between the bundle oracle's matmuls and HF's own kernels.
# Measured max ~4e-5 across the checked prompts; the floor sits well above that and far below any
# real decision margin, so the token-id assertion stays the actual gate.
HF_RTOL = 1e-3
HF_ATOL = 5e-4

# A few short ragged prompts of valid token ids, plus realistic chat prompts driven through the
# bundle tokenizer. Greedy length is bounded so the CPU gate stays cheap.
RAW_PROMPTS = ([1, 5, 9, 13, 21], [2, 7, 11], [3, 8, 16, 24, 30, 42, 7])
CHAT_PROMPTS = (
    "Write a tiny Python function that doubles an integer.",
    "Explain KV caching in one short sentence.",
    "Name two practical checks before trusting a benchmark.",
)
MAX_NEW_TOKENS = 32


def _bundle_path() -> Path:
    bundle = os.environ.get("ESME_BUNDLE_PATH")
    if bundle is None:
        pytest.skip("set ESME_BUNDLE_PATH to run the Esme HF/vLLM external-baseline parity check")
    return Path(bundle)


def _convert(bundle: Path, out: Path) -> Path:
    from scripts.convert_esme_to_hf import convert

    convert(bundle, out, max_position_embeddings=1024)
    return out


def _normalize_at_eos(generated: list[int], eos_token_ids: frozenset[int]) -> list[int]:
    out: list[int] = []
    for token in generated:
        out.append(token)
        if token in eos_token_ids:
            break
    return out


@pytest.fixture(scope="module")
def hf_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bundle = _bundle_path()
    out = tmp_path_factory.mktemp("esme-hf")
    return _convert(bundle, out)


def test_converter_emits_qwen3_tied_config(hf_checkpoint: Path) -> None:
    import json

    config = json.loads((hf_checkpoint / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["Qwen3ForCausalLM"]
    assert config["model_type"] == "qwen3"
    assert config["tie_word_embeddings"] is True
    assert config["attention_bias"] is False
    assert config["head_dim"] == 64
    assert config["num_attention_heads"] == 12
    assert config["num_key_value_heads"] == 4
    assert config["num_hidden_layers"] == 30
    assert config["hidden_size"] == 768
    assert (hf_checkpoint / "model.safetensors").is_file()
    assert (hf_checkpoint / "tokenizer.json").is_file()


def test_hf_logits_match_bundle_oracle(hf_checkpoint: Path) -> None:
    """Hard CPU gate: HF Qwen3 fp32 logits match the oracle (exact argmax, fp32-noise logits)."""
    from transformers import AutoModelForCausalLM

    oracle = PretrainBundleModel.load(_bundle_path(), dtype=torch.float32)
    hf = AutoModelForCausalLM.from_pretrained(
        hf_checkpoint, dtype=torch.float32, local_files_only=True
    ).eval()

    for prompt in RAW_PROMPTS:
        reference = oracle.logits(prompt)
        with torch.no_grad():
            hf_logits = hf(torch.tensor([prompt])).logits[0]
        assert hf_logits.shape == reference.shape
        # Token ids exact (the gate); logits within the documented fp32 floor.
        torch.testing.assert_close(hf_logits.argmax(dim=-1), reference.argmax(dim=-1))
        torch.testing.assert_close(hf_logits, reference, rtol=HF_RTOL, atol=HF_ATOL)


def test_hf_greedy_chat_matches_bundle_oracle(hf_checkpoint: Path) -> None:
    """HF greedy on real chat prompts matches the oracle greedy decode, token-for-token."""
    from transformers import AutoModelForCausalLM

    runtime = load_model_runtime("esme", bundle_path=_bundle_path(), dtype=torch.float32)
    eos = runtime.eos_token_ids
    hf = AutoModelForCausalLM.from_pretrained(
        hf_checkpoint, dtype=torch.float32, local_files_only=True
    ).eval()

    for content in CHAT_PROMPTS:
        prompt_ids = runtime.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True
        )
        reference = greedy_decode(
            runtime.model, list(prompt_ids), max_new_tokens=MAX_NEW_TOKENS, eos_token_ids=set(eos)
        )
        input_ids = torch.tensor([prompt_ids])
        with torch.no_grad():
            generated = hf.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                eos_token_id=sorted(eos),
                pad_token_id=sorted(eos)[0],
            )
        hf_generated = generated[0, input_ids.shape[1] :].tolist()
        assert _normalize_at_eos(hf_generated, eos) == _normalize_at_eos(reference, eos)


def test_three_way_routes_distinct_llm_infer_runtime(hf_checkpoint: Path) -> None:
    """The three-way seam keeps ``runtime`` as the oracle and routes llm_infer to its own runtime.

    Structural guard for the flash wiring (flash itself is GPU-only, so the GPU harness runs it on
    ``FlashAttnPagedAttention`` while CPU uses ``torch_naive``): a *distinct* llm_infer runtime must
    drive the llm_infer row, the oracle/reference stays ``runtime``, every system stays
    reference-gated, and the llm_infer mode label surfaces the engine runtime's backend class.
    """
    import math

    from llm_infer.benchmarks.esme_paged import build_requests
    from llm_infer.benchmarks.esme_three_way import run_three_way

    oracle_runtime = load_model_runtime("esme", bundle_path=_bundle_path(), dtype=torch.float32)
    engine_runtime = load_model_runtime("esme", bundle_path=_bundle_path(), dtype=torch.float32)
    assert engine_runtime is not oracle_runtime

    requests = build_requests(oracle_runtime.tokenizer, 2)
    max_new = 8
    block_size = 128
    num_blocks = (
        sum(math.ceil((len(req.prompt_ids) + max_new) / block_size) for req in requests) + 8
    )
    timings, agreements = run_three_way(
        oracle_runtime,
        hf_checkpoint,
        requests,
        max_new_tokens=max_new,
        block_size=block_size,
        num_blocks=num_blocks,
        warmup=0,
        iters=1,
        device="cpu",
        include_vllm=False,
        llm_infer_runtime=engine_runtime,
    )
    by_system = {t.system: t for t in timings}
    assert set(by_system) == {"hf_sequential", "llm_infer"}
    assert set(agreements) == {"hf_sequential", "llm_infer"}
    # CPU bundle is torch_naive; on GPU this label is FlashAttnPagedAttention. The point is the
    # label reflects the engine runtime's actual backend, so a flash row can never be mislabeled.
    backend_name = type(engine_runtime.model.backend).__name__
    assert backend_name in by_system["llm_infer"].mode
    # Both systems agree with the fp32 oracle (else no tok/s) — match-before-measuring holds. On CPU
    # (fp32 model) the agreement is exact; matches_reference is driven by the tie-tolerant profile.
    assert all(t.matches_reference for t in timings)
    assert all(agreements[s].all_ties_or_exact for s in agreements)


def test_tie_tolerant_agreement_classifies_exact_tie_and_nontie(hf_checkpoint: Path) -> None:
    """The bench agreement reuses Qwen's tie rule: exact / genuine-tie / non-tie classified."""
    from llm_infer.benchmarks.esme_paged import EsmeBenchRequest, reference_outputs
    from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement

    runtime = load_model_runtime("esme", bundle_path=_bundle_path(), dtype=torch.float32)
    eos = runtime.eos_token_ids
    prompt = runtime.tokenizer.apply_chat_template(
        [{"role": "user", "content": CHAT_PROMPTS[0]}], add_generation_prompt=True, tokenize=True
    )
    requests = [EsmeBenchRequest(request_id="r0", prompt=CHAT_PROMPTS[0], prompt_ids=tuple(prompt))]
    reference = reference_outputs(runtime.model, requests, max_new_tokens=8, eos_token_ids=eos)

    # Exact match against the oracle's own greedy decode -> agreement, zero non-tie.
    exact = tie_tolerant_agreement(runtime.model, requests, dict(reference), reference, eos)
    assert exact.exact == 1 and exact.tie == 0 and exact.nontie == 0
    assert exact.all_ties_or_exact

    # A divergence the oracle is confident about (swap the first token for a clearly-wrong one) is a
    # NON-tie -> not agreement, so the system would report no tok/s. This proves the gate still
    # rejects real divergences; it is not a blanket pass.
    gold = reference["r0"]
    wrong = [(gold[0] + 1) % runtime.model.config.vocab_size] + list(gold[1:])
    bad = tie_tolerant_agreement(runtime.model, requests, {"r0": wrong}, reference, eos)
    assert bad.nontie == 1
    assert not bad.all_ties_or_exact


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="vLLM parity needs a CUDA GPU (GPU-deferred on CPU)"
)
def test_vllm_greedy_matches_bundle_oracle(hf_checkpoint: Path) -> None:
    """vLLM greedy ids on the converted checkpoint match the oracle, token-for-token (GPU only)."""
    pytest.importorskip("vllm", reason="vLLM not installed (GPU-deferred path)")
    from vllm import LLM, SamplingParams

    runtime = load_model_runtime("esme", bundle_path=_bundle_path(), dtype=torch.float32)
    eos = runtime.eos_token_ids
    prompts = [
        runtime.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True
        )
        for content in CHAT_PROMPTS
    ]
    reference = [
        greedy_decode(
            runtime.model, list(ids), max_new_tokens=MAX_NEW_TOKENS, eos_token_ids=set(eos)
        )
        for ids in prompts
    ]

    llm = LLM(
        model=str(hf_checkpoint),
        dtype="float32",
        enable_prefix_caching=False,
        max_model_len=max(len(ids) for ids in prompts) + MAX_NEW_TOKENS,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        n=1,
        stop_token_ids=sorted(eos),
        ignore_eos=False,
    )
    results = llm.generate([{"prompt_token_ids": list(ids)} for ids in prompts], params)
    for result, gold in zip(results, reference, strict=True):
        vllm_ids = [int(token) for token in result.outputs[0].token_ids]
        assert _normalize_at_eos(vllm_ids, eos) == _normalize_at_eos(gold, eos)
