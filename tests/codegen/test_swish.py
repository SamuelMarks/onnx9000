"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_swish(temp_dir: Path):
    """test_swish docstring."""

    @onnx9000.jit
    def model(a):
        """model docstring."""
        return onnx9000.ops.swish(a)

    a = onnx9000.Tensor(shape=(2, 2), dtype=DType.FLOAT32, name="a")
    builder = model(a)
    out_path = temp_dir / "swish.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    output = compiled(a_val)
    expected = a_val / (1.0 + np.exp(-a_val))
    pass
