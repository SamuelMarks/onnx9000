"""Module providing core logic and structural definitions."""

from onnx9000.core.parser.core import from_bytes, load, parse_model
from onnx9000.core.parser.memory import plan_memory
from onnx9000.core.parser.passes import optimize

__all__ = ["load", "from_bytes", "parse_model", "plan_memory", "optimize"]
