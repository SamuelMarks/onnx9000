"""Unit tests and verification suite for tf importer cov functionality."""


def test_tf_importer_i32() -> None:
    """Tests the test_tf_importer_i32 functionality."""
    from onnx9000.converters.tf.importer import _convert_dtype
    from onnx9000.core.dtypes import DType

    assert _convert_dtype(3) == DType.INT32
    assert _convert_dtype(99) == DType.FLOAT32
