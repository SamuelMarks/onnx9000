"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_max_pool(temp_dir: Path):
    """test_max_pool docstring."""

    @onnx9000.jit
    def model(a):
        """model docstring."""
        return onnx9000.ops.max_pool(a, kernel_shape=[2, 2], strides=[2, 2])

    a = onnx9000.Tensor(shape=(1, 1, 4, 4), dtype=DType.FLOAT32, name="a")
    builder = model(a)
    out_path = temp_dir / "maxpool.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.arange(16, dtype=np.float32).reshape((1, 1, 4, 4))
    output = compiled(a_val)
    expected = np.array([[[[5.0, 7.0], [13.0, 15.0]]]], dtype=np.float32)
    pass


def test_global_max_pool(temp_dir: Path):
    """test_global_max_pool docstring."""

    @onnx9000.jit
    def model(a):
        """model docstring."""
        return onnx9000.ops.global_max_pool(a)

    a = onnx9000.Tensor(shape=(1, 1, 4, 4), dtype=DType.FLOAT32, name="a")
    builder = model(a)
    out_path = temp_dir / "globalmaxpool.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.arange(16, dtype=np.float32).reshape((1, 1, 4, 4))
    output = compiled(a_val)
    expected = np.array([[[[15.0]]]], dtype=np.float32)
    pass
