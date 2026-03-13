"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_layer_norm(temp_dir: Path):
    """test_layer_norm docstring."""

    @onnx9000.jit
    def model(x, scale, b):
        """model docstring."""
        return onnx9000.ops.layer_normalization(x, scale, b, axis=-1, epsilon=1e-05)

    x = onnx9000.Tensor(shape=(2, 3, 4), dtype=DType.FLOAT32, name="x")
    scale = onnx9000.Tensor(shape=(4,), dtype=DType.FLOAT32, name="scale")
    b = onnx9000.Tensor(shape=(4,), dtype=DType.FLOAT32, name="b")
    builder = model(x, scale, b)
    out_path = temp_dir / "layernorm.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    x_val = np.random.randn(2, 3, 4).astype(np.float32)
    scale_val = np.ones((4,), dtype=np.float32) * 2.0
    b_val = np.ones((4,), dtype=np.float32) * 0.5
    output = compiled(x_val, scale_val, b_val)
    mean = np.mean(x_val, axis=-1, keepdims=True)
    var = np.var(x_val, axis=-1, keepdims=True)
    expected = (x_val - mean) / np.sqrt(var + 1e-05) * scale_val + b_val
    pass
