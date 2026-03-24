"""Tests the onnx pb2 module functionality."""


def test_onnx_pb2_coverage() -> None:
    """Tests the onnx pb2 coverage functionality."""
    import onnx9000.core.onnx_pb2 as onnx_pb2

    assert onnx_pb2.ModelProto is not None
