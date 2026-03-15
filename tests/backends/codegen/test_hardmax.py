"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_hardmax(temp_dir: Path):
    """Tests the test hardmax functionality."""

    @onnx9000.jit
    def model(a):
        """Provides model functionality and verification."""
        return onnx9000.core.ops.hardmax(a, axis=1)

    a = onnx9000.Tensor(shape=(2, 3), dtype=DType.FLOAT32, name="a")
    builder = model(a)
    out_path = temp_dir / "hardmax.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.array([[1.0, 3.0, 2.0], [6.0, 5.0, 4.0]], dtype=np.float32)
    output = compiled(a_val)
    expected = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    pass
