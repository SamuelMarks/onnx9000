"""Unit tests and verification suite for genai functionality."""

from onnx9000_genai import run


def test_run() -> None:
    """Verify that run behaves as expected."""
    assert run() == "[genai] processed"
