"""Unit tests for onnx9000_triton_server main entrypoint."""

from onnx9000_triton_server import TritonServer


def test_process() -> None:
    """Verify TritonServer().process() via onnx9000_triton_server."""
    assert TritonServer().process("test") == "Triton Server processed test"
