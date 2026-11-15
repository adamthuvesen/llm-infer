# llm-infer

Terms for the `llm-infer` serving engine. Bundle and chat terms come from `esme-pretrain`.

## Contract terms (owned by esme-pretrain)

Each term below: "Defined in esme-pretrain `GLOSSARY.md`."

- **Bundle**, **Bundle format**, **Manifest**, **Base bundle / chat bundle**
- **Chat template** and **Chat template version**. This repo only renders it. It does not write it.
- **EOS (`<eos>`)**, **EOS ids**, **Special tokens**, **Extension fields**
- **add_special_tokens**. The load rules are in `docs/adr/0001-eos-and-special-token-rules.md`.

## Serving terms

**Paged KV cache**:
K/V rows in fixed-size physical blocks. Each request has a `BlockTable` from token position to block. K/V is stored after RoPE and before GQA expansion (`llm_infer/kv_cache/paged_kv_cache.py`).
_Avoid_: "the cache" for the prefix cache

**Prefix cache**:
Reuse of complete prompt blocks across requests, shared by refcount (`llm_infer/kv_cache/prefix_cache.py`). A write into a shared partial block copies first.

**Continuous batching**:
Requests enter and leave the running batch at decode-step boundaries. The batch is not fixed for a request set.

**Request state**:
One of `Pending`, `Prefilling`, `DecodeReady`, `Terminal` (`llm_infer/request/request.py`). A `DecodeReady` request always has a block table.

**Preemption**:
The scheduler returns a running request to `Pending`. A decoding request gets prefill mode `recompute`. A prefilling request keeps its mode. The request keeps its generated tokens and RNG, and its K/V is recomputed on return (`llm_infer/request/request.py`).

**Decode window**:
A run of decode steps whose tokens stay on the device. EOS is checked only at flush. A request can decode throwaway tokens after its EOS. They are dropped, never emitted (`llm_infer/serving/decode_window.py::DecodeWindow`).

**default_add_special_tokens**:
A `TokenizersJsonTokenizer` argument in `llm_infer/model/runtime.py`. It holds the resolved `add_special_tokens` value. It is not the manifest key `decoding.default_add_special_tokens`, which this repo never reads.

**Reference**:
The full-recompute `PretrainBundleModel.logits()` path for Esme, and pinned HuggingFace output for Qwen. Fast paths are checked against it.

**headline_eligible**:
True only when the reference status is `exact` or `accepted_numerical`. Public tok/s needs it. Raw timing is kept in every row (`llm_infer/benchmarks/reference_policy.py`).
_Avoid_: "verified" for a row that only has A/B parity

**A/B parity**:
Direct candidate-versus-baseline agreement. It is separate from reference status.

## Name drift

- `default_add_special_tokens` names two things (see above).
