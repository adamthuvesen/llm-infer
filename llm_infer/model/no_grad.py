"""Typed ``torch.no_grad()`` decorator: torch's own annotation erases the wrapped signature."""

from collections.abc import Callable
from typing import cast

import torch


def no_grad[F: Callable[..., object]](function: F) -> F:
    return cast(F, torch.no_grad()(function))
