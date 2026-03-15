"""Module providing core logic and structural definitions."""


def test_tf_importer_i32():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.tf.importer import _convert_dtype
    from onnx9000.core.dtypes import DType

    assert _convert_dtype(3) == DType.INT32
    assert _convert_dtype(99) == DType.FLOAT32
