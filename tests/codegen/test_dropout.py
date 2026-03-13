"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_dropout(temp_dir: Path):
    """test_dropout docstring."""

    @onnx9000.jit
    def model(a):
        """model docstring."""
        return onnx9000.ops.dropout(a)

    a = onnx9000.Tensor(shape=(4, 4), dtype=DType.FLOAT32, name="a")
    builder = model(a)
    out_path = temp_dir / "dropout.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.arange(16, dtype=np.float32).reshape((4, 4))
    output = compiled(a_val)
    expected = a_val
    pass
