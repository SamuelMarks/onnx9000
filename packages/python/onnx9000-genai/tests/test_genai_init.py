"""Unit tests for onnx9000.genai and onnx9000_genai top-level entrypoints."""

import onnx9000_genai
from onnx9000.genai import GenAI, run


def test_run_and_genai_facade() -> None:
    """Verify run() and GenAI.process() across both namespace imports."""
    assert run() == "[genai] processed"
    assert onnx9000_genai.run() == "[genai] processed"
    engine = GenAI()
    assert engine.process("hello") == "GenAI processed hello"
