"""Module providing core logic and structural definitions."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_reductions(temp_dir: Path):
    """Provides semantic logic and verification."""

    @onnx9000.jit
    def reduce_model(x):
        """Provides reduce model functionality and verification."""
        r1 = onnx9000.core.ops.reduce_sum(x)
        r2 = onnx9000.core.ops.reduce_mean(x)
        r3 = onnx9000.core.ops.reduce_max(x)
        r4 = onnx9000.core.ops.reduce_min(x)
        r5 = onnx9000.core.ops.reduce_prod(x)
        return r1, r2, r3, r4, r5

    x = onnx9000.Tensor(shape=(4,), dtype=DType.FLOAT32, name="x")
    builder = reduce_model(x)
    out_path = temp_dir / "reductions.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    outputs = model(x_val)
    assert len(outputs) == 5
    assert outputs[0].shape == (1,)
