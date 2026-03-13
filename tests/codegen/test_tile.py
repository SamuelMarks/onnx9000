"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_tile(temp_dir: Path):
    """test_tile docstring."""

    @onnx9000.jit
    def model(a):
        """model docstring."""
        return onnx9000.ops.tile(a, np.array([2, 3], dtype=np.int64))

    a = onnx9000.Tensor(shape=(2, 2), dtype=DType.FLOAT32, name="a")
    builder = model(a)
    out_path = temp_dir / "tile.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.arange(4, dtype=np.float32).reshape((2, 2))
    output = compiled(a_val)
    expected = np.tile(a_val, (2, 3))
    pass
