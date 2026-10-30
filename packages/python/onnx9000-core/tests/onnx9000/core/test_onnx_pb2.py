"""Unit tests and verification suite for ONNX protobuf definitions."""

import onnx9000.core.onnx_pb2


def test_module_load() -> None:
    """Verify that the generated onnx_pb2 protobuf module loads cleanly."""
    assert onnx9000.core.onnx_pb2 is not None
