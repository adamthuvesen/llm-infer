"""Packaged reference and benchmark input fixtures."""

from importlib.resources import files

_FIXTURES = files(__name__)

QWEN_COT_GOLDEN = _FIXTURES / "qwen2_5_coder_3b_instruct_cot.json"
