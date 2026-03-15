"""Module providing core logic and structural definitions."""


def test_proto_utils_str_shape():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.tensor import Tensor
    from onnx9000.export.proto_utils import to_value_info_proto
    from onnx9000.core.dtypes import DType

    t = Tensor(shape=("batch_size", 3), dtype=DType.FLOAT32, name="t")
    proto = to_value_info_proto(t)
    assert proto.type.tensor_type.shape.dim[0].dim_param == "batch_size"
    assert proto.type.tensor_type.shape.dim[1].dim_value == 3
