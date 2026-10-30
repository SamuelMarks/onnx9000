"""Unit tests for onnx9000_gguf re-exports."""

from onnx9000_gguf import GGUF, GGUFError, GGUFParser, parse_gguf, run


def test_gguf_reexports() -> None:
    """Verify all symbols are re-exported from onnx9000_gguf."""
    assert run() == "[gguf] processed"
    assert GGUF is not None
    assert GGUFError is not None
    assert GGUFParser is not None
    assert callable(parse_gguf)
