"""Tests the onnx pb2 module functionality."""


def test_onnx_pb2_coverage() -> None:
    """Tests the onnx pb2 coverage functionality."""
    from onnx9000.core import onnx_pb2

    assert onnx_pb2.ModelProto is not None
