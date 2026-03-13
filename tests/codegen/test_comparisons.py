"""Module docstring."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType
import pytest


def test_comparisons(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def model(a, b):
        """model docstring."""
        le = onnx9000.ops.less_or_equal(a, b)
        ge = onnx9000.ops.greater_or_equal(a, b)
        return le, ge

    a = onnx9000.Tensor(shape=(4,), dtype=DType.FLOAT32, name="a")
    b = onnx9000.Tensor(shape=(4,), dtype=DType.FLOAT32, name="b")
    builder = model(a, b)
    out_path = temp_dir / "comparisons.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    a_v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b_v = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    outputs = model(a_v, b_v)
    np.testing.assert_array_equal(outputs[0], np.array([True, True, False, False]))
    np.testing.assert_array_equal(outputs[1], np.array([False, True, True, True]))
