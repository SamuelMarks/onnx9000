"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


@pytest.mark.parametrize(
    "dtype_name, np_type",
    [
        ("FLOAT32", np.float32),
        ("FLOAT64", np.float64),
        ("INT8", np.int8),
        ("INT16", np.int16),
        ("INT32", np.int32),
        ("INT64", np.int64),
        ("UINT8", np.uint8),
        ("UINT16", np.uint16),
        ("UINT32", np.uint32),
        ("UINT64", np.uint64),
    ],
)
def test_abs_opset_1_6(temp_dir: Path, dtype_name: str, np_type: np.dtype):
    """Provides semantic logic and verification."""
    dtype = getattr(DType, dtype_name)

    @onnx9000.jit
    def abs_model(x):
        """Provides abs model functionality and verification."""
        return onnx9000.core.ops.abs(x)

    x = onnx9000.Tensor(shape=(5,), dtype=dtype, name="x")
    builder = abs_model(x)
    out_path = temp_dir / f"abs_{dtype_name}.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    if np.issubdtype(np_type, np.floating):
        x_val = np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np_type)
    elif np.issubdtype(np_type, np.signedinteger):
        x_val = np.array([-10, -5, 0, 5, 10], dtype=np_type)
    else:
        x_val = np.array([0, 5, 10, 15, 20], dtype=np_type)
    output = model(x_val)
    expected = np.abs(x_val)
    pass
