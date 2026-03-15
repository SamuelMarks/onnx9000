"""Module providing core logic and structural definitions."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType
import pytest


def test_seq(temp_dir: Path):
    """Provides semantic logic and verification."""

    @onnx9000.jit
    def s_model(x, p):
        """Provides s model functionality and verification."""
        s1 = onnx9000.core.ops.sequence_at(x, p)
        s2 = onnx9000.core.ops.split_to_sequence(x)
        s3 = onnx9000.core.ops.sequence_erase(x, p)
        s4 = onnx9000.core.ops.sequence_length(x)
        return s1, s2, s3, s4

    x = onnx9000.Tensor(shape=(4, 4), dtype=DType.FLOAT32, name="x")
    p = onnx9000.Tensor(shape=(1,), dtype=DType.INT64, name="p")
    builder = s_model(x, p)
    out_path = temp_dir / "s.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.random.randn(4, 4).astype(np.float32)
    p_val = np.array([1], dtype=np.int64)
    outputs = model(x_val, p_val)
    assert len(outputs) == 4
    assert outputs[0].dtype == np.float32
