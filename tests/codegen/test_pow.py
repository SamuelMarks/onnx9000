"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_pow(temp_dir: Path):
    """test_pow docstring."""

    @onnx9000.jit
    def model(a, b):
        """model docstring."""
        return onnx9000.ops.pow(a, b)

    a = onnx9000.Tensor(shape=(4, 3), dtype=DType.FLOAT32, name="a")
    b = onnx9000.Tensor(shape=(4, 3), dtype=DType.FLOAT32, name="b")
    builder = model(a, b)
    out_path = temp_dir / "pow.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.array([[2.0] * 3] * 4, dtype=np.float32)
    b_val = np.array([[3.0] * 3] * 4, dtype=np.float32)
    output = compiled(a_val, b_val)
    expected = np.power(a_val, b_val)
    pass
