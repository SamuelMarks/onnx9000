"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_cumsum(temp_dir: Path):
    """test_cumsum docstring."""

    @onnx9000.jit
    def model(a, axis):
        """model docstring."""
        return onnx9000.ops.cumsum(a, axis, exclusive=0, reverse=0)

    a = onnx9000.Tensor(shape=(3, 4), dtype=DType.FLOAT32, name="a")
    axis = onnx9000.Tensor(shape=(1,), dtype=DType.INT64, name="axis")
    builder = model(a, axis)
    out_path = temp_dir / "cumsum.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.arange(12, dtype=np.float32).reshape((3, 4))
    axis_val = np.array([1], dtype=np.int64)
    output = compiled(a_val, axis_val)
    expected = np.cumsum(a_val, axis=1)
    pass
