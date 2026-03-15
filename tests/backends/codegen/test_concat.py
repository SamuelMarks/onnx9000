"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_concat_opsets(temp_dir: Path):
    """Tests the test concat opsets functionality."""

    @onnx9000.jit
    def concat_model(a, b, c):
        """Provides concat model functionality and verification."""
        return onnx9000.core.ops.concat([a, b, c], axis=1)

    a = onnx9000.Tensor(shape=(2, 2), dtype=DType.FLOAT32, name="a")
    b = onnx9000.Tensor(shape=(2, 3), dtype=DType.FLOAT32, name="b")
    c = onnx9000.Tensor(shape=(2, 1), dtype=DType.FLOAT32, name="c")
    builder = concat_model(a, b, c)
    out_path = temp_dir / "concat.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    a_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_val = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32)
    c_val = np.array([[11.0], [12.0]], dtype=np.float32)
    output = model(a_val, b_val, c_val)
    expected = np.concatenate([a_val, b_val, c_val], axis=1)
    pass
