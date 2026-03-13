"""Module docstring."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType
import pytest


def test_shrink_size(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def ss_model(x):
        """ss_model docstring."""
        s = onnx9000.ops.shrink(x)
        sz = onnx9000.ops.size(x)
        return s, sz

    x = onnx9000.Tensor(shape=(4, 4), dtype=DType.FLOAT32, name="x")
    builder = ss_model(x)
    out_path = temp_dir / "ss.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.random.randn(4, 4).astype(np.float32)
    outputs = model(x_val)
    assert len(outputs) == 2
    assert outputs[0].dtype == np.float32
    assert outputs[1].dtype == np.int64
