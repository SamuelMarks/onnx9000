"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_sum(temp_dir: Path):
    """Tests the test sum functionality."""

    @onnx9000.jit
    def model(a, b, c):
        """Provides model functionality and verification."""
        return onnx9000.core.ops.sum([a, b, c])

    a = onnx9000.Tensor(shape=(2, 2), dtype=DType.FLOAT32, name="a")
    b = onnx9000.Tensor(shape=(2, 2), dtype=DType.FLOAT32, name="b")
    c = onnx9000.Tensor(shape=(2, 2), dtype=DType.FLOAT32, name="c")
    builder = model(a, b, c)
    out_path = temp_dir / "sum.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_val = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    c_val = np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32)
    output = compiled(a_val, b_val, c_val)
    expected = a_val + b_val + c_val
    pass
