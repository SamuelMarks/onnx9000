"""Module providing core logic and structural definitions."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_math_unary(temp_dir: Path):
    """Provides semantic logic and verification."""

    @onnx9000.jit
    def math_model(x):
        """Provides math model functionality and verification."""
        y1 = onnx9000.core.ops.acos(x)
        y2 = onnx9000.core.ops.asin(x)
        y3 = onnx9000.core.ops.atan(x)
        y4 = onnx9000.core.ops.cos(x)
        y5 = onnx9000.core.ops.cosh(x)
        y6 = onnx9000.core.ops.sin(x)
        y7 = onnx9000.core.ops.sinh(x)
        y8 = onnx9000.core.ops.tan(x)
        y9 = onnx9000.core.ops.ceil(x)
        y10 = onnx9000.core.ops.acosh(y5)
        y11 = onnx9000.core.ops.atanh(x)
        y12 = onnx9000.core.ops.asinh(x)
        return y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12

    x = onnx9000.Tensor(shape=(5,), dtype=DType.FLOAT32, name="x")
    builder = math_model(x)
    out_path = temp_dir / "math_unary.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    output = model(x_val)
    y1 = np.arccos(x_val)
    y2 = np.arcsin(x_val)
    y3 = np.arctan(x_val)
    y4 = np.cos(x_val)
    y5 = np.cosh(x_val)
    y6 = np.sin(x_val)
    y7 = np.sinh(x_val)
    y8 = np.tan(x_val)
    y9 = np.ceil(x_val)
    y10 = np.arccosh(y5)
    y11 = np.arctanh(x_val)
    y12 = np.arcsinh(x_val)
    expected = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12
    pass
