"""Read nested JSON-shaped results (``dict[str, object]``) by key path."""

from __future__ import annotations


def dig(value: object, *path: str | int) -> object:
    for key in path:
        if isinstance(key, int):
            assert isinstance(value, list), f"{key!r}: expected a list, got {type(value).__name__}"
            value = value[key]
        else:
            assert isinstance(value, dict), f"{key!r}: expected a dict, got {type(value).__name__}"
            value = value[key]
    return value


def dig_dict(value: object, *path: str | int) -> dict[str, object]:
    found = dig(value, *path)
    assert isinstance(found, dict), f"{path}: expected a dict, got {type(found).__name__}"
    return found


def dig_number(value: object, *path: str | int) -> float:
    number = dig(value, *path)
    assert isinstance(number, int | float), f"{path}: expected a number, got {number!r}"
    return number


def dig_rows(value: object, *path: str | int) -> list[dict[str, object]]:
    rows = dig(value, *path)
    assert isinstance(rows, list)
    return [row for row in rows if isinstance(row, dict)]
