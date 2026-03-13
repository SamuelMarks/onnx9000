"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_dft(temp_dir: Path):
    """test_dft docstring."""

    @onnx9000.jit
    def model(a):
        """model docstring."""
        return onnx9000.ops.dft(a, axis=1)

    a = onnx9000.Tensor(shape=(3, 4, 2), dtype=DType.FLOAT32, name="a")
    builder = model(a)
    out_path = temp_dir / "dft.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.ones((3, 4, 2), dtype=np.float32)
    output = compiled(a_val)
    expected = np.zeros((3, 4, 2), dtype=np.float32)
    pass
