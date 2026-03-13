"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_trilu(temp_dir: Path):
    """test_trilu docstring."""

    @onnx9000.jit
    def model(a):
        """model docstring."""
        return onnx9000.ops.trilu(a, upper=1)

    a = onnx9000.Tensor(shape=(3, 3), dtype=DType.FLOAT32, name="a")
    builder = model(a)
    out_path = temp_dir / "trilu.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.ones((3, 3), dtype=np.float32)
    output = compiled(a_val)
    expected = np.triu(a_val)
    pass
