"""Fail a Modal GPU run loudly when it stops making progress.

A run that hangs inside CUDA, a lock, or a network read prints nothing and keeps billing until the
function timeout. On 2026-10-24 two runs sat silent for 50 minutes that way. ``arm`` dumps every
thread's stack and exits the process once ``limit_s`` passes without a ``progress()`` call, so a
stall fails the Modal call within minutes and leaves its stacks in the log.
"""

from __future__ import annotations

import faulthandler
import sys

# Longest legitimate gap between progress calls: model load, graph capture, FlashInfer JIT, and a
# torch.compile warmup each take a few minutes at most.
DEFAULT_LIMIT_S = 15 * 60

_limit_s: float | None = None


def arm(limit_s: float = DEFAULT_LIMIT_S) -> None:
    global _limit_s
    _limit_s = limit_s
    faulthandler.dump_traceback_later(limit_s, exit=True, file=sys.stderr)


def progress() -> None:
    """Restart the stall deadline; call after every row or batch the run completes."""
    if _limit_s is not None:
        faulthandler.dump_traceback_later(_limit_s, exit=True, file=sys.stderr)
