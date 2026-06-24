"""Freeze the 8-prompt llm-rlvr-sql GRPO rollout slice as a committed, reproducible fixture.

Phase E's rollout-timing comparison replays llm-rlvr-sql's *actual* GRPO rollout shape, not a
synthetic microbench. One GRPO rollout batch is 8 prompts × G=4 = 32 completions
(``per_device_batch_size=8``, ``num_generations=4`` in llm-rlvr-sql's GrpoHyperparams). The
prompts must be byte-identical to llm-rlvr-sql's eval path, so this builds them through
llm-rlvr-sql's OWN builders — ``build_messages(style='cot')`` + ``serialize_schema`` — then
applies the pinned Qwen Instruct chat template (``add_generation_prompt=True``), exactly as
the eval/transformers path does. The resulting prompt token ids are frozen here so the
benchmark replays them with no llm-rlvr-sql dependency and no re-tokenization drift.

This runs locally, once. It needs llm-rlvr-sql's builders (pure pydantic) and ``datasets`` for
the pinned Spider-dev questions. From the llm-infer repo root:

    uv run --with datasets --with pydantic python scripts/build_rollout_fixture.py

Inputs (override via env): ``RLVR_SQL_ROOT`` (llm-rlvr-sql checkout, read-only),
``TEXT2SQL_SPIDER_ROOT`` (an extracted ``spider_data/`` for ``tables.json`` — llm-rlvr-sql's
local copy by default, so no multi-GB archive download). Writes
``tests/fixtures/rollout_grpo_s0_spider_dev.json`` and prints its sha256.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig

LLM_INFER_ROOT = Path(__file__).resolve().parents[1]
RLVR_SQL_ROOT = Path(
    os.environ.get("RLVR_SQL_ROOT", Path.home() / "dev" / "menti" / "llm-rlvr-sql")
)
SPIDER_ROOT = Path(
    os.environ.get("TEXT2SQL_SPIDER_ROOT", RLVR_SQL_ROOT / "data" / "spider" / "spider_data")
)

# llm-rlvr-sql GRPO rollout shape (llm_rlvr_sql.train.grpo.GrpoHyperparams) — anchor is grpo-s0.
NUM_PROMPTS = 8  # per_device_batch_size
NUM_GENERATIONS = 4  # G
MAX_COMPLETION_LENGTH = 1024
TEMPERATURE = 1.0
TOP_P = 1.0
SAMPLING_SEED = 0  # GrpoHyperparams.seed
SELECTION_SEED = 0  # frozen, reproducible choice of the 8 Spider-dev prompts
PROMPT_STYLE = "cot"  # the RL model's chain-of-thought instruction

FIXTURE_PATH = LLM_INFER_ROOT / "tests" / "fixtures" / "rollout_grpo_s0_spider_dev.json"


def _load_build_messages(rlvr_root: Path):
    """Load llm-rlvr-sql's ``build_messages`` from prompt.py without importing the eval package."""
    prompt_path = rlvr_root / "src" / "llm_rlvr_sql" / "eval" / "prompt.py"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"llm-rlvr-sql prompt builder not found at {prompt_path}")
    spec = importlib.util.spec_from_file_location("rlvr_eval_prompt", prompt_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_messages


def _eos_token_ids(model_id: str, revision: str, tokenizer: object) -> list[int]:
    """The generation-stopping ids HF uses: generation-config eos plus the tokenizer eos."""
    gen_config = GenerationConfig.from_pretrained(model_id, revision=revision)
    ids: set[int] = set()
    cfg_eos = gen_config.eos_token_id
    if isinstance(cfg_eos, int):
        ids.add(cfg_eos)
    elif cfg_eos is not None:
        ids.update(cfg_eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    sys.path.insert(0, str(RLVR_SQL_ROOT / "src"))
    sys.path.insert(0, str(LLM_INFER_ROOT))

    from llm_rlvr_sql.data.artifacts import SPIDER_HF_REPO, SPIDER_HF_REVISION
    from llm_rlvr_sql.data.schema import serialize_schema
    from llm_rlvr_sql.data.spider import load_examples, load_schemas

    from llm_infer.model.config import MODEL_ID, MODEL_REVISION

    # Load build_messages straight from its file: importing llm_rlvr_sql.eval as a package runs
    # eval/__init__.py, which pulls the SQL sandbox/verifier chain (sqlparse). prompt.py itself
    # only depends on the light data layer, so executing it directly keeps this builder lean.
    build_messages = _load_build_messages(RLVR_SQL_ROOT)

    tables_json = SPIDER_ROOT / "tables.json"
    if not tables_json.is_file():
        raise FileNotFoundError(
            f"no tables.json at {tables_json}; set TEXT2SQL_SPIDER_ROOT to an extracted "
            "spider_data/ dir (llm-rlvr-sql's data/spider/spider_data by default)."
        )
    schemas = load_schemas(tables_json)
    examples = load_examples("dev")  # HF xlangai/spider @ pinned revision — the eval source
    if len(examples) < NUM_PROMPTS:
        raise ValueError(f"need {NUM_PROMPTS} prompts; Spider dev has only {len(examples)}")

    # Frozen, reproducible slice: a seeded sample over the dev split, sorted for legibility.
    chosen_indices = sorted(random.Random(SELECTION_SEED).sample(range(len(examples)), NUM_PROMPTS))
    selected = [examples[i] for i in chosen_indices]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    eos_token_ids = _eos_token_ids(MODEL_ID, MODEL_REVISION, tokenizer)

    prompts = []
    for position, (dev_index, example) in enumerate(zip(chosen_indices, selected, strict=True)):
        if example.db_id not in schemas:
            raise KeyError(f"schema for db_id={example.db_id!r} missing from {tables_json}")
        messages = build_messages(
            serialize_schema(schemas[example.db_id]),
            example.question,
            exemplars=(),  # zero-shot, matching the grpo-s0 rollout prompt
            evidence=example.evidence,
            style=PROMPT_STYLE,
        )
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        prompt_ids = inputs["input_ids"][0].tolist()
        prompts.append(
            {
                "prompt_id": f"p{position}",
                "dev_index": dev_index,
                "db_id": example.db_id,
                "question": example.question,
                "num_prompt_tokens": len(prompt_ids),
                "prompt_ids": prompt_ids,
                "prompt_text": tokenizer.decode(prompt_ids),
            }
        )

    fixture = {
        "format_version": 1,
        "description": (
            "One frozen rlvr-sql GRPO rollout batch (grpo-s0 shape): 8 Spider-dev prompts × "
            "G=4 = 32 completions. Prompts built through rlvr-sql build_messages(cot) + "
            "serialize_schema, tokenized with the pinned Qwen Instruct chat template."
        ),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "dataset": {
            "name": "spider",
            "split": "dev (validation)",
            "hf_repo": SPIDER_HF_REPO,
            "hf_revision": SPIDER_HF_REVISION,
            "tables_json_source": "spider_data/tables.json",
            "tables_json_sha256": _sha256_file(tables_json),
        },
        "prompt_builder": {
            "source": "rlvr-sql eval/prompt.py build_messages(style='cot'), exemplars=()",
            "schema": "rlvr-sql data/schema.py serialize_schema",
            "chat_template": "tokenizer.apply_chat_template(add_generation_prompt=True)",
            "style": PROMPT_STYLE,
            "few_shot_k": 0,
        },
        "selection": {"seed": SELECTION_SEED, "dev_indices": chosen_indices},
        "rollout": {
            "anchor": "grpo-s0",
            "num_prompts": NUM_PROMPTS,
            "num_generations": NUM_GENERATIONS,
            "completions_per_batch": NUM_PROMPTS * NUM_GENERATIONS,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "sampling_seed": SAMPLING_SEED,
            "eos_token_ids": eos_token_ids,
        },
        "prompts": prompts,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
    FIXTURE_PATH.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    token_counts = [p["num_prompt_tokens"] for p in prompts]
    print(f"wrote {len(prompts)} prompts to {FIXTURE_PATH}")
    print(f"dev_indices={chosen_indices}  db_ids={[p['db_id'] for p in prompts]}")
    print(f"prompt token counts: {token_counts} (min {min(token_counts)}, max {max(token_counts)})")
    print(f"eos_token_ids={eos_token_ids}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
