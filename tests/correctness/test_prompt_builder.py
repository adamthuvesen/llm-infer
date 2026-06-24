"""Pin the replicated llm-rlvr-sql prompt builder against its exact expected bytes.

``tests/correctness/prompt.py`` is a verbatim copy of llm-rlvr-sql's ``cot`` path (it
must not import llm-rlvr-sql). These tests freeze the exact output so any drift in the
copy breaks the build instead of silently desynchronizing the goldens. The expected
strings below were produced by the real llm-rlvr-sql functions and compared byte-for-byte
at fixture-creation time.
"""

from __future__ import annotations

from tests.correctness.prompt import (
    COT_INSTRUCTION,
    Column,
    DatabaseSchema,
    ForeignKey,
    Table,
    build_cot_messages,
    serialize_schema,
)

_SCHEMA = DatabaseSchema(
    db_id="store",
    tables=(
        Table(
            name="customer",
            columns=(Column(name="id", type="int"), Column(name="name", type="text")),
            primary_key=("id",),
        ),
        Table(
            name="orders",
            columns=(Column(name="id", type="int"), Column(name="customer_id", type="int")),
            primary_key=("id",),
        ),
    ),
    foreign_keys=(
        ForeignKey(
            from_table="orders", from_column="customer_id", to_table="customer", to_column="id"
        ),
    ),
)

_EXPECTED_SCHEMA_TEXT = (
    'CREATE TABLE "customer" (\n'
    '  "id" int,\n'
    '  "name" text,\n'
    '  PRIMARY KEY ("id")\n'
    ");\n"
    "\n"
    'CREATE TABLE "orders" (\n'
    '  "id" int,\n'
    '  "customer_id" int,\n'
    '  PRIMARY KEY ("id"),\n'
    '  FOREIGN KEY ("customer_id") REFERENCES "customer"("id")\n'
    ");"
)


def test_serialize_schema_is_byte_exact() -> None:
    assert serialize_schema(_SCHEMA) == _EXPECTED_SCHEMA_TEXT


def test_cot_instruction_is_exact() -> None:
    assert COT_INSTRUCTION == (
        "You are an expert data analyst. Given a SQLite database schema and a question, write a "
        "single SQLite query that answers it. Reason step by step inside <think></think>, then "
        "give the final query in a ```sql ... ``` code block."
    )


def test_build_cot_messages_layout() -> None:
    messages = build_cot_messages(_SCHEMA, "How many customers placed an order?")
    assert messages == [
        {"role": "system", "content": COT_INSTRUCTION},
        {
            "role": "user",
            "content": (
                f"Database schema:\n{_EXPECTED_SCHEMA_TEXT}\n\n"
                "Question: How many customers placed an order?"
            ),
        },
    ]


def test_evidence_is_appended() -> None:
    messages = build_cot_messages(_SCHEMA, "q?", evidence="a hint")
    assert messages[1]["content"].endswith("Question: q?\nEvidence: a hint")
