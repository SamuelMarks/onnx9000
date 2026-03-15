"""Module providing core logic and structural definitions."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_ternary_fast(temp_dir: Path):
    """Provides semantic logic and verification."""

    @onnx9000.jit
    def test_model(cond, x, y):
        """Provides semantic logic and verification."""
        w = onnx9000.core.ops.where(cond, x, y)
        return w

    cond = onnx9000.Tensor(shape=(4,), dtype=DType.BOOL, name="cond")
    x = onnx9000.Tensor(shape=(4,), dtype=DType.FLOAT32, name="x")
    y = onnx9000.Tensor(shape=(4,), dtype=DType.FLOAT32, name="y")
    builder = test_model(cond, x, y)
    out_path = temp_dir / "where.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    cond_val = np.array([True, False, True, False], dtype=np.bool_)
    x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y_val = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    outputs = model(cond_val, x_val, y_val)
    w_exp = np.where(cond_val, x_val, y_val)
    pass
