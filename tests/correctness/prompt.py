"""The llm-rlvr prompt builder, replicated byte-for-byte.

The oracle is only meaningful if it decodes the *exact* prompts llm-rlvr feeds the
model. This is a verbatim copy of llm-rlvr's ``cot`` path — the ``cot`` system
instruction, ``serialize_schema``'s ``CREATE TABLE`` blocks, and ``build_messages``'s
user-content layout — pinned here so the engine has no dependency on the llm-rlvr
package. It is a copy on purpose, not an import: the goldens are frozen against these
exact bytes, and a unit test asserts they still match.

Source (read-only, llm-rlvr @ Jun 2026):
  src/llm_rlvr/data/schema.py :: serialize_schema
  src/llm_rlvr/eval/prompt.py :: INSTRUCTION ("cot"), _user_content, build_messages
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Verbatim llm-rlvr "cot" system instruction (eval/prompt.py :: INSTRUCTION).
COT_INSTRUCTION = (
    "You are an expert data analyst. Given a SQLite database schema and a question, write a "
    "single SQLite query that answers it. Reason step by step inside <think></think>, then give "
    "the final query in a ```sql ... ``` code block."
)


@dataclass(frozen=True)
class Column:
    name: str
    type: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass(frozen=True)
class DatabaseSchema:
    db_id: str
    tables: tuple[Table, ...]
    foreign_keys: tuple[ForeignKey, ...] = field(default_factory=tuple)


def serialize_schema(schema: DatabaseSchema) -> str:
    """Verbatim llm-rlvr ``serialize_schema`` (data/schema.py).

    Deterministic by construction.
    """
    fks_by_table: dict[str, list[str]] = {}
    for fk in schema.foreign_keys:
        line = f'  FOREIGN KEY ("{fk.from_column}") REFERENCES "{fk.to_table}"("{fk.to_column}")'
        fks_by_table.setdefault(fk.from_table, []).append(line)

    blocks: list[str] = []
    for table in schema.tables:
        lines = [f'CREATE TABLE "{table.name}" (']
        body = [f'  "{col.name}" {col.type}' for col in table.columns]
        if table.primary_key:
            pk = ", ".join(f'"{name}"' for name in table.primary_key)
            body.append(f"  PRIMARY KEY ({pk})")
        body.extend(fks_by_table.get(table.name, []))
        lines.append(",\n".join(body))
        lines.append(");")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _user_content(schema_text: str, question: str, evidence: str | None = None) -> str:
    """Verbatim llm-rlvr ``_user_content`` (eval/prompt.py)."""
    content = f"Database schema:\n{schema_text}\n\nQuestion: {question}"
    if evidence:
        content += f"\nEvidence: {evidence}"
    return content


def build_cot_messages(
    schema: DatabaseSchema, question: str, evidence: str | None = None
) -> list[dict[str, str]]:
    """The llm-rlvr ``cot`` chat messages with no few-shot exemplars.

    Mirrors ``build_messages(..., style="cot")`` with ``exemplars=()`` — the system
    instruction followed by the single target user turn. oracle fixtures use the
    zero-shot path; few-shot exemplars are a later concern.
    """
    return [
        {"role": "system", "content": COT_INSTRUCTION},
        {"role": "user", "content": _user_content(serialize_schema(schema), question, evidence)},
    ]
