"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_bitwise_opsets(temp_dir: Path):
    """Provides semantic logic and verification."""

    @onnx9000.jit
    def bitwise_model(a, b, c):
        """Provides bitwise model functionality and verification."""
        ba = onnx9000.core.ops.bitwise_and(a, b)
        bn = onnx9000.core.ops.bitwise_not(a)
        bo = onnx9000.core.ops.bitwise_or(a, b)
        bx = onnx9000.core.ops.bitwise_xor(a, b)
        bsl = onnx9000.core.ops.bitshift(c, c, "LEFT")
        bsr = onnx9000.core.ops.bitshift(c, c, "RIGHT")
        return ba, bn, bo, bx, bsl, bsr

    a = onnx9000.Tensor(shape=(3,), dtype=DType.INT32, name="a")
    b = onnx9000.Tensor(shape=(3,), dtype=DType.INT32, name="b")
    c = onnx9000.Tensor(shape=(3,), dtype=DType.UINT32, name="c")
    builder = bitwise_model(a, b, c)
    out_path = temp_dir / "bitwise.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    a_val = np.array([1, 2, 3], dtype=np.int32)
    b_val = np.array([3, 2, 1], dtype=np.int32)
    c_val = np.array([1, 2, 3], dtype=np.uint32)
    outputs = model(a_val, b_val, c_val)
    ba_exp = np.bitwise_and(a_val, b_val)
    bn_exp = np.bitwise_not(a_val)
    bo_exp = np.bitwise_or(a_val, b_val)
    bx_exp = np.bitwise_xor(a_val, b_val)
    bsl_exp = np.left_shift(c_val, c_val)
    bsr_exp = np.right_shift(c_val, c_val)
    np.testing.assert_array_equal(outputs[0], ba_exp)
    np.testing.assert_array_equal(outputs[1], bn_exp)
    np.testing.assert_array_equal(outputs[2], bo_exp)
    np.testing.assert_array_equal(outputs[3], bx_exp)
    np.testing.assert_array_equal(outputs[4], bsl_exp)
    np.testing.assert_array_equal(outputs[5], bsr_exp)
