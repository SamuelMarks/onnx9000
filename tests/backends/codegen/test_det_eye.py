"""Module providing core logic and structural definitions."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType
import pytest


def test_det_eye(temp_dir: Path):
    """Provides semantic logic and verification."""

    @onnx9000.jit
    def d_model(x):
        """Provides d model functionality and verification."""
        d = onnx9000.core.ops.det(x)
        e = onnx9000.core.ops.eye_like(x)
        return d, e

    x = onnx9000.Tensor(shape=(3, 3), dtype=DType.FLOAT32, name="x")
    builder = d_model(x)
    out_path = temp_dir / "de.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.random.randn(3, 3).astype(np.float32)
    outputs = model(x_val)
    assert len(outputs) == 2
    assert outputs[0].dtype == np.float32
    assert outputs[1].dtype == np.float32
