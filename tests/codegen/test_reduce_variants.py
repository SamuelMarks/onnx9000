"""Module docstring."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType
import pytest


def test_reduce_variants(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def rv_model(x):
        """rv_model docstring."""
        ss = onnx9000.ops.reduce_sum_square(x)
        l1 = onnx9000.ops.reduce_l1(x)
        l2 = onnx9000.ops.reduce_l2(x)
        ls = onnx9000.ops.reduce_log_sum(x)
        lse = onnx9000.ops.reduce_log_sum_exp(x)
        return ss, l1, l2, ls, lse

    x = onnx9000.Tensor(shape=(3, 3), dtype=DType.FLOAT32, name="x")
    builder = rv_model(x)
    out_path = temp_dir / "rv.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.random.randn(3, 3).astype(np.float32)
    outputs = model(x_val)
    assert len(outputs) == 5
    for o in outputs:
        assert o.dtype == np.float32
