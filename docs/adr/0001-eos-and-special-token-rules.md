# EOS and special-token rules at bundle load

Status: accepted

A mismatch between training text and served text is silent. It ships a model that behaves worse with no error. So the loader fails loud on any ambiguous EOS or tokenizer setting. The code is `llm_infer/model/runtime.py::_dense_eos_ids` and `llm_infer/model/runtime.py::_dense_add_special_tokens`.

- **EOS ids.** The loader reads `eos_token_ids` and `decoding.eos_token_ids`. If both exist and differ, load fails. A chat bundle with no EOS ids fails. A base bundle with none uses the tokenizer `<eos>` id. With no `<eos>` token, load fails.
- **add_special_tokens.** The loader takes the first set value from `tokenizer.add_special_tokens`, then top-level `add_special_tokens`, then `chat_template.add_special_tokens`. The value must be a boolean. With none set, it uses `true`.
- **Chat rendering.** Served text uses the manifest `chat_template`. A v2 assistant turn ends with the text `<eos>`. This matches training only if `tokenizer.json` registers `<eos>` as a special token.

## Consequences

A bundle that disagrees with itself cannot be served. The producer rules are in the esme-pretrain ADR on chat-template versions. Tests: `tests/model/test_runtime.py::test_esme_v2_chat_template_ends_each_served_assistant_turn_with_the_eos_id`.
