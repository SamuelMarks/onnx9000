"""Module providing core logic and structural definitions."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType
import pytest


def test_nms_nz(temp_dir: Path):
    """Provides semantic logic and verification."""

    @onnx9000.jit
    def nms_model(boxes, scores):
        """Provides nms model functionality and verification."""
        nms = onnx9000.core.ops.non_max_suppression(boxes, scores)
        nz = onnx9000.core.ops.non_zero(scores)
        return nms, nz

    boxes = onnx9000.Tensor(shape=(1, 5, 4), dtype=DType.FLOAT32, name="boxes")
    scores = onnx9000.Tensor(shape=(1, 1, 5), dtype=DType.FLOAT32, name="scores")
    builder = nms_model(boxes, scores)
    out_path = temp_dir / "nms.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    b_val = np.random.randn(1, 5, 4).astype(np.float32)
    s_val = np.random.randn(1, 1, 5).astype(np.float32)
    outputs = model(b_val, s_val)
    assert len(outputs) == 2
    assert outputs[0].dtype == np.int64
    assert outputs[1].dtype == np.int64
