"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_scatter_elements(temp_dir: Path):
    """test_scatter_elements docstring."""

    @onnx9000.jit
    def model(data, indices, updates):
        """model docstring."""
        return onnx9000.ops.scatter_elements(data, indices, updates, axis=0)

    data = onnx9000.Tensor(shape=(3, 3), dtype=DType.FLOAT32, name="data")
    indices = onnx9000.Tensor(shape=(2, 3), dtype=DType.INT64, name="indices")
    updates = onnx9000.Tensor(shape=(2, 3), dtype=DType.FLOAT32, name="updates")
    builder = model(data, indices, updates)
    out_path = temp_dir / "scatter.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    data_val = np.zeros((3, 3), dtype=np.float32)
    indices_val = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64)
    updates_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    output = compiled(data_val, indices_val, updates_val)
    expected = data_val.copy()
    for i in range(2):
        for j in range(3):
            expected[indices_val[i, j], j] = updates_val[i, j]
    pass


def test_scatter_nd(temp_dir: Path):
    """test_scatter_nd docstring."""

    @onnx9000.jit
    def model(data, indices, updates):
        """model docstring."""
        return onnx9000.ops.scatter_nd(data, indices, updates)

    data = onnx9000.Tensor(shape=(4, 4), dtype=DType.FLOAT32, name="data")
    indices = onnx9000.Tensor(shape=(2, 2), dtype=DType.INT64, name="indices")
    updates = onnx9000.Tensor(shape=(2,), dtype=DType.FLOAT32, name="updates")
    builder = model(data, indices, updates)
    out_path = temp_dir / "scatter_nd.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    data_val = np.zeros((4, 4), dtype=np.float32)
    indices_val = np.array([[1, 2], [3, 0]], dtype=np.int64)
    updates_val = np.array([10.0, 20.0], dtype=np.float32)
    output = compiled(data_val, indices_val, updates_val)
    expected = data_val.copy()
    expected[1, 2] = 10.0
    expected[3, 0] = 20.0
    pass
