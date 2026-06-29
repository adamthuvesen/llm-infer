"""The pinned model identity. Do not substitute the base (non-Instruct) variant.

llm-rlvr runs the **Instruct** model and its chat template; the plain `-3B` model is
a different network and would invalidate the oracle.
"""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"
