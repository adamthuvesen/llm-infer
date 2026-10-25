"""``torch.compile`` versions of the bundle model's RMSNorm and RoPE.

Each function is a chain of small elementwise kernels. Compiled, each becomes one or two fused
Triton kernels, which cut 37-44% of a full-depth graphed decode step (docs/findings.md entry 23).
The bundle model and the decode graphs look ``rms_norm`` and ``apply_rope`` up as module globals
at call time, so the switch below routes every later call and graph capture through the chosen
version. The switch is process-wide. Graphs keep the kernels they captured, so switch before
building the engine whose graphs should use them.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

import llm_infer.model.decode_graph as decode_graph_module
import llm_infer.model.pretrain_bundle as bundle_module
from llm_infer.model import rope_utils

_Versions = tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]
_compiled: _Versions | None = None


def use_fused_elementwise(enabled: bool) -> None:
    """Route RMSNorm and RoPE through the compiled versions, or back to the originals.

    Compilation happens on the first call of each shape, not here. A graph capture's warmup
    runs every segment before recording it, so compilation finishes before capture starts.
    """
    global _compiled
    if enabled:
        if _compiled is None:
            _compiled = (
                torch.compile(rope_utils.rms_norm, dynamic=True),
                torch.compile(rope_utils.apply_rope, dynamic=True),
            )
        rms_norm, apply_rope = _compiled
    else:
        rms_norm, apply_rope = rope_utils.rms_norm, rope_utils.apply_rope
    bundle_module.rms_norm = rms_norm
    bundle_module.apply_rope = apply_rope
    decode_graph_module.rms_norm = rms_norm
