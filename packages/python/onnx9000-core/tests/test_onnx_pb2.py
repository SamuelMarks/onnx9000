import importlib
from unittest.mock import patch


def test_onnx_pb2_coverage() -> None:
    import onnx9000.core.onnx_pb2 as onnx_pb2
    from google.protobuf import descriptor

    with patch.object(descriptor, "_USE_C_DESCRIPTORS", False):
        importlib.reload(onnx_pb2)
