"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_batchnorm_opsets(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def batchnorm_model(x, scale, b, mean, var):
        """batchnorm_model docstring."""
        from onnx9000.frontend.utils import record_op

        if hasattr(onnx9000.ops, "batch_normalization"):
            return onnx9000.ops.batch_normalization(x, scale, b, mean, var)
        else:
            return record_op(
                "BatchNormalization", [x, scale, b, mean, var], {"epsilon": 1e-05}
            )

    x = onnx9000.Tensor(shape=(1, 2, 1, 3), dtype=DType.FLOAT32, name="x")
    scale = onnx9000.Tensor(shape=(2,), dtype=DType.FLOAT32, name="scale")
    b = onnx9000.Tensor(shape=(2,), dtype=DType.FLOAT32, name="b")
    mean = onnx9000.Tensor(shape=(2,), dtype=DType.FLOAT32, name="mean")
    var = onnx9000.Tensor(shape=(2,), dtype=DType.FLOAT32, name="var")
    builder = batchnorm_model(x, scale, b, mean, var)
    out_path = temp_dir / "batchnorm.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.random.randn(1, 2, 1, 3).astype(np.float32)
    scale_val = np.array([1.0, 2.0], dtype=np.float32)
    b_val = np.array([0.1, 0.2], dtype=np.float32)
    mean_val = np.array([0.0, 1.0], dtype=np.float32)
    var_val = np.array([1.0, 0.5], dtype=np.float32)
    output = model(x_val, scale_val, b_val, mean_val, var_val)
    epsilon = 1e-05
    expected = np.zeros_like(x_val)
    for c in range(2):
        expected[0, c, 0, :] = (x_val[0, c, 0, :] - mean_val[c]) / np.sqrt(
            var_val[c] + epsilon
        ) * scale_val[c] + b_val[c]
    pass
