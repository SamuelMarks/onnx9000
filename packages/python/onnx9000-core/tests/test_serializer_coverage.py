"""Tests the serializer coverage module functionality."""

from onnx9000.core.serializer import _serialize_shape


def test_serialize_str_dim() -> None:
    """Tests the serialize str dim functionality."""
    shape = ("N",)
    proto = _serialize_shape(shape)
    assert proto.dim[0].dim_param == "N"
