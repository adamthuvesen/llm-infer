"""Correctness checks for the esme-pretrain dense export-bundle bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_infer.model.decode import greedy_decode
from llm_infer.model.pretrain_bundle import (
    BUNDLE_FORMAT,
    PretrainBundleError,
    PretrainBundleModel,
)

VOCAB_SIZE = 11
HIDDEN_SIZE = 4
INTERMEDIATE_SIZE = 8
NUM_LAYERS = 2
NUM_HEADS = 2
NUM_KV_HEADS = 1


def _write_tiny_bundle(
    root: Path,
    *,
    qk_norm: bool = False,
    logit_soft_cap: float | None = 7.5,
) -> Path:
    bundle = root / "tiny_pretrain_bundle"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.mkdir(exist_ok=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "format": BUNDLE_FORMAT,
                "model": {"name": "tiny-dense"},
                "tokenizer": {"path": "tokenizer.json", "format": "tokenizers-json"},
            }
        ),
        encoding="utf-8",
    )
    (bundle / "config.json").write_text(
        json.dumps(
            {
                "vocab_size": VOCAB_SIZE,
                "embedding_dim": HIDDEN_SIZE,
                "feedforward_dim": INTERMEDIATE_SIZE,
                "layers": NUM_LAYERS,
                "heads": NUM_HEADS,
                "kv_heads": NUM_KV_HEADS,
                "norm_eps": 1e-5,
                "rope_theta": 10_000.0,
                "tie_embeddings": True,
                "qk_norm": qk_norm,
                "z_loss_weight": 1e-4,
            }
        ),
        encoding="utf-8",
    )
    if logit_soft_cap is not None:
        config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        config["logit_soft_cap"] = logit_soft_cap
        (bundle / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (bundle / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {
                    "type": "WordLevel",
                    "vocab": {f"tok_{idx}": idx for idx in range(VOCAB_SIZE)},
                    "unk_token": "tok_0",
                },
                "pre_tokenizer": {"type": "Whitespace"},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "format_version": 1,
            "metadata": {"key_format": BUNDLE_FORMAT},
            "state_dict": _tiny_state_dict(qk_norm=qk_norm),
        },
        bundle / "weights.pt",
    )
    return bundle


def _tiny_state_dict(*, qk_norm: bool = False) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260626)

    def randn(*shape: int, scale: float = 0.12) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32) * scale

    weights: dict[str, torch.Tensor] = {
        "token_embedding.weight": randn(VOCAB_SIZE, HIDDEN_SIZE),
        "final_norm.weight": torch.tensor([0.91, 1.03, 0.97, 1.11], dtype=torch.float32),
    }
    head_dim = HIDDEN_SIZE // NUM_HEADS
    for layer in range(NUM_LAYERS):
        prefix = f"blocks.{layer}."
        weights[prefix + "attention_norm.weight"] = torch.ones(HIDDEN_SIZE) + randn(
            HIDDEN_SIZE, scale=0.03
        )
        weights[prefix + "feedforward_norm.weight"] = torch.ones(HIDDEN_SIZE) + randn(
            HIDDEN_SIZE, scale=0.03
        )
        weights[prefix + "attention.wq.weight"] = randn(NUM_HEADS * head_dim, HIDDEN_SIZE)
        weights[prefix + "attention.wk.weight"] = randn(NUM_KV_HEADS * head_dim, HIDDEN_SIZE)
        weights[prefix + "attention.wv.weight"] = randn(NUM_KV_HEADS * head_dim, HIDDEN_SIZE)
        weights[prefix + "attention.wo.weight"] = randn(HIDDEN_SIZE, HIDDEN_SIZE)
        if qk_norm:
            weights[prefix + "attention.q_norm.weight"] = torch.tensor([0.77, 1.23])
            weights[prefix + "attention.k_norm.weight"] = torch.tensor([1.17, 0.83])
        weights[prefix + "feedforward.w_gate.weight"] = randn(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        weights[prefix + "feedforward.w_up.weight"] = randn(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        weights[prefix + "feedforward.w_down.weight"] = randn(HIDDEN_SIZE, INTERMEDIATE_SIZE)
    return weights


def _reference_logits(
    token_ids: list[int], state_dict: dict[str, torch.Tensor], *, qk_norm: bool
) -> torch.Tensor:
    ids = torch.tensor(token_ids, dtype=torch.long)
    hidden = state_dict["token_embedding.weight"][ids]
    cos, sin = _reference_rope_tables(len(token_ids))
    head_dim = HIDDEN_SIZE // NUM_HEADS

    for layer in range(NUM_LAYERS):
        prefix = f"blocks.{layer}."
        residual = hidden
        x = _reference_rms_norm(hidden, state_dict[prefix + "attention_norm.weight"])
        q = (x @ state_dict[prefix + "attention.wq.weight"].T).view(
            len(token_ids), NUM_HEADS, head_dim
        )
        k = (x @ state_dict[prefix + "attention.wk.weight"].T).view(
            len(token_ids), NUM_KV_HEADS, head_dim
        )
        v = (x @ state_dict[prefix + "attention.wv.weight"].T).view(
            len(token_ids), NUM_KV_HEADS, head_dim
        )
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        if qk_norm:
            q = _reference_rms_norm(q, state_dict[prefix + "attention.q_norm.weight"])
            k = _reference_rms_norm(k, state_dict[prefix + "attention.k_norm.weight"])
        q = _reference_apply_rope(q, cos, sin)
        k = _reference_apply_rope(k, cos, sin)
        k = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=0)
        v = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=0)
        attn = _reference_attention(q, k, v)
        merged = attn.transpose(0, 1).reshape(len(token_ids), HIDDEN_SIZE)
        hidden = residual + merged @ state_dict[prefix + "attention.wo.weight"].T

        residual = hidden
        x = _reference_rms_norm(hidden, state_dict[prefix + "feedforward_norm.weight"])
        gate = x @ state_dict[prefix + "feedforward.w_gate.weight"].T
        up = x @ state_dict[prefix + "feedforward.w_up.weight"].T
        hidden = (
            residual
            + (torch.nn.functional.silu(gate) * up)
            @ state_dict[prefix + "feedforward.w_down.weight"].T
        )

    hidden = _reference_rms_norm(hidden, state_dict["final_norm.weight"])
    return hidden @ state_dict["token_embedding.weight"].T


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    return (weight.float() * xf).to(x.dtype)


def _reference_rope_tables(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = HIDDEN_SIZE // NUM_HEADS
    half = head_dim // 2
    positions = torch.arange(seq_len, dtype=torch.float32)
    inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _reference_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(x.dtype).unsqueeze(0)
    sin = sin.to(x.dtype).unsqueeze(0)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def _reference_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    q_len = query.shape[1]
    kv_len = key.shape[1]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    col = torch.arange(kv_len).view(1, kv_len)
    row = torch.arange(q_len).view(q_len, 1)
    scores = scores.masked_fill(col > row, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), value.float()).to(query.dtype)


def test_loads_bundle_manifest_config_weights_and_tokenizer(tmp_path: Path) -> None:
    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path))

    assert model.config.hidden_size == HIDDEN_SIZE
    assert model.config.intermediate_size == INTERMEDIATE_SIZE
    assert model.config.num_hidden_layers == NUM_LAYERS
    assert model.config.num_attention_heads == NUM_HEADS
    assert model.config.num_key_value_heads == NUM_KV_HEADS
    assert model.config.tie_word_embeddings is True
    assert model.config.logit_soft_cap == 7.5
    assert model.tokenizer_path.name == "tokenizer.json"
    assert model.tie_word_embeddings is True
    assert model.config.qk_norm is False
    assert "lm_head.weight" not in model.w


def test_dense_bundle_runs_through_engine_cached_contract(tmp_path: Path) -> None:
    from llm_infer.serving import InferenceEngine, Request

    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path))
    engine = InferenceEngine(model, block_size=8, num_blocks=32)
    engine.add_request(Request("dense-req", [1, 4, 7], max_new_tokens=3, eos_token_ids=frozenset()))

    outputs = engine.run()

    assert len(outputs["dense-req"]) == 3
    assert all(isinstance(token, int) for token in outputs["dense-req"])


def test_dense_bundle_chunked_prefill_uses_same_logits(tmp_path: Path) -> None:
    from llm_infer.kv_cache.paged_kv_cache import PagedKVCache

    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path))
    cache = PagedKVCache(
        num_layers=model.num_layers,
        num_blocks=16,
        block_size=4,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        dtype=model.dtype,
    )
    table = cache.new_request()

    first = model.prefill_chunk([1, 4, 7], cache, table, start_pos=0, chunk_size=2)
    final = model.prefill_chunk([1, 4, 7], cache, table, start_pos=2, chunk_size=2)

    assert first.shape == (VOCAB_SIZE,)
    torch.testing.assert_close(final, model.logits([1, 4, 7])[-1])


def test_qk_norm_loads_and_affects_logits(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path, qk_norm=True, logit_soft_cap=None)
    state_dict = torch.load(bundle / "weights.pt", weights_only=True)["state_dict"]
    model = PretrainBundleModel.load(bundle)

    logits = model.logits([1, 4, 7])
    expected = _reference_logits([1, 4, 7], state_dict, qk_norm=True)
    without_qk_norm = _reference_logits([1, 4, 7], state_dict, qk_norm=False)

    assert model.config.qk_norm is True
    assert "layers.0.attn.q_norm.weight" in model.w
    assert "layers.0.attn.k_norm.weight" in model.w
    torch.testing.assert_close(logits, expected, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(logits, without_qk_norm)


def test_qk_norm_requires_exported_norm_weights(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path, qk_norm=True)
    state_dict = _tiny_state_dict(qk_norm=True)
    del state_dict["blocks.0.attention.q_norm.weight"]
    torch.save(
        {"metadata": {"key_format": BUNDLE_FORMAT}, "state_dict": state_dict},
        bundle / "weights.pt",
    )

    with pytest.raises(PretrainBundleError, match="q_norm"):
        PretrainBundleModel.load(bundle)


def test_allows_disabled_logit_soft_cap_alias(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    del config["logit_soft_cap"]
    config["final_logit_softcapping"] = 0.0
    (bundle / "config.json").write_text(json.dumps(config), encoding="utf-8")

    model = PretrainBundleModel.load(bundle)

    assert model.config.logit_soft_cap == 0.0


def test_rejects_negative_logit_soft_cap(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    config["logit_soft_cap"] = -0.1
    (bundle / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(PretrainBundleError, match="non-negative"):
        PretrainBundleModel.load(bundle)


def test_rejects_bad_manifest_format(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    (bundle / "manifest.json").write_text(json.dumps({"format": "other"}), encoding="utf-8")

    with pytest.raises(PretrainBundleError, match=BUNDLE_FORMAT):
        PretrainBundleModel.load(bundle)


def test_rejects_tokenizer_paths_outside_bundle(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    (bundle / "manifest.json").write_text(
        json.dumps({"format": BUNDLE_FORMAT, "tokenizer": {"path": "../tokenizer.json"}}),
        encoding="utf-8",
    )

    with pytest.raises(PretrainBundleError, match="inside the bundle"):
        PretrainBundleModel.load(bundle)


def test_rejects_weights_without_dense_key_metadata(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    torch.save({"state_dict": _tiny_state_dict()}, bundle / "weights.pt")

    with pytest.raises(PretrainBundleError, match="metadata"):
        PretrainBundleModel.load(bundle)


def test_rejects_malformed_weight_shapes(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    state_dict = _tiny_state_dict()
    state_dict["blocks.0.attention.wq.weight"] = torch.zeros(3, HIDDEN_SIZE)
    torch.save(
        {"metadata": {"key_format": BUNDLE_FORMAT}, "state_dict": state_dict},
        bundle / "weights.pt",
    )

    with pytest.raises(PretrainBundleError, match="attention.wq"):
        PretrainBundleModel.load(bundle)


def test_logits_match_tiny_golden_tensor(tmp_path: Path) -> None:
    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path, logit_soft_cap=None))

    logits = model.logits([1, 4, 7])

    assert logits.shape == (3, VOCAB_SIZE)
    expected_last = torch.tensor(
        [
            0.19731820,
            0.09545554,
            0.13589795,
            0.16449249,
            -0.03189749,
            -0.16612272,
            -0.52944112,
            0.33006054,
            -0.22187583,
            0.06887253,
            0.24853352,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(logits[-1], expected_last, rtol=1e-6, atol=1e-6)


def test_greedy_generation_for_token_ids(tmp_path: Path) -> None:
    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path))

    assert greedy_decode(model, [1, 4, 7], max_new_tokens=4, eos_token_ids=set()) == [
        7,
        7,
        7,
        7,
    ]


def test_qwen_loader_import_still_available() -> None:
    from llm_infer.model.qwen import QwenModel

    assert QwenModel.__name__ == "QwenModel"
