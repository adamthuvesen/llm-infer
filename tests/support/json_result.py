"""Read nested JSON-shaped results (``dict[str, object]``) by key path."""

from __future__ import annotations


def dig(value: object, *path: str) -> object:
    for key in path:
        assert isinstance(value, dict), f"{key!r}: expected a dict, got {type(value).__name__}"
        value = value[key]
    return value


def dig_number(value: object, *path: str) -> float:
    number = dig(value, *path)
    assert isinstance(number, int | float), f"{path}: expected a number, got {number!r}"
    return number


def dig_rows(value: object, *path: str) -> list[dict[str, object]]:
    rows = dig(value, *path)
    assert isinstance(rows, list)
    return [row for row in rows if isinstance(row, dict)]  # pyright: ignore[reportUnknownVariableType]
