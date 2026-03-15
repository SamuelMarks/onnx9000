"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_gather_elements(temp_dir: Path):
    """Tests the test gather elements functionality."""

    @onnx9000.jit
    def model(data, indices):
        """Provides model functionality and verification."""
        return onnx9000.core.ops.gather_elements(data, indices, axis=1)

    data = onnx9000.Tensor(shape=(2, 2), dtype=DType.FLOAT32, name="data")
    indices = onnx9000.Tensor(shape=(2, 2), dtype=DType.INT64, name="indices")
    builder = model(data, indices)
    out_path = temp_dir / "gather_elements.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    data_val = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices_val = np.array([[0, 0], [1, 0]], dtype=np.int64)
    output = compiled(data_val, indices_val)
    expected = np.array([[1, 1], [4, 3]], dtype=np.float32)
    pass
