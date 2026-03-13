"""Module docstring."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType
import pytest


def test_constant_ops(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def const_model(x):
        """const_model docstring."""
        c1 = onnx9000.ops.constant(value=1.0)
        c2 = onnx9000.ops.constant_of_shape(x, value=0.0)
        return c1, c2

    x = onnx9000.Tensor(shape=(2,), dtype=DType.INT64, name="x")
    builder = const_model(x)
    out_path = temp_dir / "constants.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.array([2, 3], dtype=np.int64)
    outputs = model(x_val)
    assert len(outputs) == 2
    assert outputs[0].shape == (1,)
    assert outputs[0].dtype == np.float32
