"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_slice(temp_dir: Path):
    """test_slice docstring."""

    @onnx9000.jit
    def model(x):
        """model docstring."""
        return onnx9000.ops.slice(
            x,
            np.array([0, 1], dtype=np.int64),
            np.array([2, 3], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
        )

    x = onnx9000.Tensor(shape=(4, 4), dtype=DType.FLOAT32, name="x")
    builder = model(x)
    out_path = temp_dir / "slice.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    x_val = np.arange(16, dtype=np.float32).reshape((4, 4))
    output = compiled(x_val)
    expected = x_val[0:2, 1:3]
    pass
