"""The golden cases: a few short text-to-SQL prompts kept tiny so the oracle runs on CPU.

Each case is a (schema, question) pair in the llm-rlvr shape. Adding a case here and
re-running ``scripts/generate_goldens.py`` regenerates the committed fixture. Keep
schemas small and ``max_new_tokens`` short — the point is a fast, exact oracle, not
coverage of long generations.
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.correctness.prompt import Column, DatabaseSchema, ForeignKey, Table


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    schema: DatabaseSchema
    question: str


GOLDEN_CASES: tuple[GoldenCase, ...] = (
    GoldenCase(
        case_id="single_table_count",
        schema=DatabaseSchema(
            db_id="library",
            tables=(
                Table(
                    name="book",
                    columns=(
                        Column(name="id", type="int"),
                        Column(name="title", type="text"),
                        Column(name="year", type="int"),
                    ),
                    primary_key=("id",),
                ),
            ),
        ),
        question="How many books were published after 2000?",
    ),
    GoldenCase(
        case_id="two_table_join",
        schema=DatabaseSchema(
            db_id="store",
            tables=(
                Table(
                    name="customer",
                    columns=(
                        Column(name="id", type="int"),
                        Column(name="name", type="text"),
                    ),
                    primary_key=("id",),
                ),
                Table(
                    name="orders",
                    columns=(
                        Column(name="id", type="int"),
                        Column(name="customer_id", type="int"),
                        Column(name="total", type="real"),
                    ),
                    primary_key=("id",),
                ),
            ),
            foreign_keys=(
                ForeignKey(
                    from_table="orders",
                    from_column="customer_id",
                    to_table="customer",
                    to_column="id",
                ),
            ),
        ),
        question="What is the name of the customer with the highest total order value?",
    ),
    GoldenCase(
        case_id="single_table_group_by",
        schema=DatabaseSchema(
            db_id="hr",
            tables=(
                Table(
                    name="employee",
                    columns=(
                        Column(name="id", type="int"),
                        Column(name="department", type="text"),
                        Column(name="salary", type="int"),
                    ),
                    primary_key=("id",),
                ),
            ),
        ),
        question="What is the average salary per department?",
    ),
)
