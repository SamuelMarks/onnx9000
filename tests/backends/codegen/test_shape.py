"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_reshape(temp_dir: Path):
    """Tests the test reshape functionality."""

    @onnx9000.jit
    def model(a):
        """Provides model functionality and verification."""
        return onnx9000.core.ops.reshape(a, np.array([2, -1], dtype=np.int64))

    a = onnx9000.Tensor(shape=(4, 3), dtype=DType.FLOAT32, name="a")
    builder = model(a)
    out_path = temp_dir / "reshape.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.arange(12, dtype=np.float32).reshape((4, 3))
    output = compiled(a_val)
    expected = a_val.reshape((2, -1))
    pass
