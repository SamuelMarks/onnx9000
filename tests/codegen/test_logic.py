"""Module docstring."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_logic_ops(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def logic_model(x, y, cond):
        """logic_model docstring."""
        e = onnx9000.ops.equal(x, y)
        g = onnx9000.ops.greater(x, y)
        l = onnx9000.ops.less(x, y)
        a = onnx9000.ops.and_(cond, cond)
        o = onnx9000.ops.or_(cond, cond)
        n = onnx9000.ops.not_(cond)
        return e, g, l, a, o, n

    x = onnx9000.Tensor(shape=(4,), dtype=DType.FLOAT32, name="x")
    y = onnx9000.Tensor(shape=(4,), dtype=DType.FLOAT32, name="y")
    cond = onnx9000.Tensor(shape=(4,), dtype=DType.BOOL, name="cond")
    builder = logic_model(x, y, cond)
    out_path = temp_dir / "logic.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y_val = np.array([4.0, 3.0, 3.0, 1.0], dtype=np.float32)
    cond_val = np.array([True, False, True, False], dtype=np.bool_)
    outputs = model(x_val, y_val, cond_val)
    e_exp = x_val == y_val
    g_exp = x_val > y_val
    l_exp = x_val < y_val
    a_exp = cond_val & cond_val
    o_exp = cond_val | cond_val
    n_exp = ~cond_val
    np.testing.assert_array_equal(outputs[0], e_exp)
    np.testing.assert_array_equal(outputs[1], g_exp)
    np.testing.assert_array_equal(outputs[2], l_exp)
    np.testing.assert_array_equal(outputs[3], a_exp)
    np.testing.assert_array_equal(outputs[4], o_exp)
    np.testing.assert_array_equal(outputs[5], n_exp)
