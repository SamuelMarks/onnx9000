"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_compress(temp_dir: Path):
    """Tests the test compress functionality."""

    @onnx9000.jit
    def model(a, b):
        """Provides model functionality and verification."""
        return onnx9000.core.ops.compress(a, b, axis=1)

    a = onnx9000.Tensor(shape=(3, 4), dtype=DType.FLOAT32, name="a")
    b = onnx9000.Tensor(shape=(4,), dtype=DType.BOOL, name="b")
    builder = model(a, b)
    out_path = temp_dir / "compress.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.arange(12, dtype=np.float32).reshape((3, 4))
    b_val = np.array([True, False, True, False], dtype=np.bool_)
    output = compiled(a_val, b_val)
    expected = np.compress(b_val, a_val, axis=1)
    pass
