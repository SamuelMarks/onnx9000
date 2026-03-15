"""Module providing core logic and structural definitions."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType
import pytest


def test_lp_ops(temp_dir: Path):
    """Provides semantic logic and verification."""

    @onnx9000.jit
    def lp_model(x):
        """Provides lp model functionality and verification."""
        norm = onnx9000.core.ops.lp_normalization(x, axis=1)
        pool = onnx9000.core.ops.lp_pool(x, kernel_shape=[2, 2])
        return norm, pool

    x = onnx9000.Tensor(shape=(1, 1, 4, 4), dtype=DType.FLOAT32, name="x")
    builder = lp_model(x)
    out_path = temp_dir / "lp.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.random.randn(1, 1, 4, 4).astype(np.float32)
    outputs = model(x_val)
    assert len(outputs) == 2
    assert outputs[0].dtype == np.float32
    assert outputs[1].dtype == np.float32
