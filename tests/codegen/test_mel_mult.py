"""Module docstring."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType
import pytest


def test_mel_mult(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def mm_model(m, d, s, l, u, inp):
        """mm_model docstring."""
        mel = onnx9000.ops.mel_weight_matrix(m, d, s, l, u)
        mult = onnx9000.ops.multinomial(inp)
        return mel, mult

    m = onnx9000.Tensor(shape=(1,), dtype=DType.INT64, name="m")
    d = onnx9000.Tensor(shape=(1,), dtype=DType.INT64, name="d")
    s = onnx9000.Tensor(shape=(1,), dtype=DType.INT64, name="s")
    l = onnx9000.Tensor(shape=(1,), dtype=DType.FLOAT32, name="l")
    u = onnx9000.Tensor(shape=(1,), dtype=DType.FLOAT32, name="u")
    inp = onnx9000.Tensor(shape=(2, 3), dtype=DType.FLOAT32, name="inp")
    builder = mm_model(m, d, s, l, u, inp)
    out_path = temp_dir / "mm.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    m_v = np.array([40], dtype=np.int64)
    d_v = np.array([400], dtype=np.int64)
    s_v = np.array([16000], dtype=np.int64)
    l_v = np.array([20.0], dtype=np.float32)
    u_v = np.array([8000.0], dtype=np.float32)
    i_v = np.random.randn(2, 3).astype(np.float32)
    outputs = model(m_v, d_v, s_v, l_v, u_v, i_v)
    assert len(outputs) == 2
    assert outputs[0].dtype == np.float32
    assert outputs[1].dtype == np.int32
