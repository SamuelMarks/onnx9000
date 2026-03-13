"""Module docstring."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType
import pytest


def test_rev_scatter(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def rs_model(x, lengths, indices, updates):
        """rs_model docstring."""
        r = onnx9000.ops.reverse_sequence(x, lengths)
        s = onnx9000.ops.scatter(x, indices, updates)
        se = onnx9000.ops.scatter_elements(x, indices, updates)
        snd = onnx9000.ops.scatter_nd(x, indices, updates)
        return r, s, se, snd

    x = onnx9000.Tensor(shape=(4, 4), dtype=DType.FLOAT32, name="x")
    lengths = onnx9000.Tensor(shape=(4,), dtype=DType.INT64, name="lengths")
    indices = onnx9000.Tensor(shape=(2, 2), dtype=DType.INT64, name="indices")
    updates = onnx9000.Tensor(shape=(2, 2), dtype=DType.FLOAT32, name="updates")
    builder = rs_model(x, lengths, indices, updates)
    out_path = temp_dir / "rs.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.random.randn(4, 4).astype(np.float32)
    l_val = np.array([1, 2, 3, 4], dtype=np.int64)
    i_val = np.array([[0, 1], [1, 0]], dtype=np.int64)
    u_val = np.random.randn(2, 2).astype(np.float32)
    outputs = model(x_val, l_val, i_val, u_val)
    assert len(outputs) == 4
    for o in outputs:
        assert o.dtype == np.float32
