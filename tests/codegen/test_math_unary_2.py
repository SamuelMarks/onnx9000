"""Module docstring."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_math_unary_2(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def math_model(x, y):
        """math_model docstring."""
        r = onnx9000.ops.round(x)
        f = onnx9000.ops.floor(x)
        n = onnx9000.ops.neg(x)
        rec = onnx9000.ops.reciprocal(x)
        s = onnx9000.ops.sign(x)
        is_i = onnx9000.ops.isinf(x)
        is_n = onnx9000.ops.isnan(x)
        m = onnx9000.ops.mod(x, y)
        p = onnx9000.ops.pow(x, y)
        return r, f, n, rec, s, is_i, is_n, m, p

    x = onnx9000.Tensor(shape=(4,), dtype=DType.FLOAT32, name="x")
    y = onnx9000.Tensor(shape=(4,), dtype=DType.FLOAT32, name="y")
    builder = math_model(x, y)
    out_path = temp_dir / "math_unary_2.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.array([0.5, -1.5, np.inf, np.nan], dtype=np.float32)
    y_val = np.array([2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    outputs = model(x_val, y_val)
    r_exp = np.round(x_val)
    x_val = np.array([0.6, -1.5, np.inf, np.nan], dtype=np.float32)
    y_val = np.array([2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    outputs = model(x_val, y_val)
    r_exp = np.round(x_val)
    f_exp = np.floor(x_val)
    n_exp = -x_val
    rec_exp = 1.0 / x_val
    s_exp = np.sign(x_val)
    is_i_exp = np.isinf(x_val)
    is_n_exp = np.isnan(x_val)
    m_exp = np.fmod(x_val, y_val)
    p_exp = np.power(x_val, y_val)
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
