"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_depthtospace(temp_dir: Path):
    """Tests the test depthtospace functionality."""

    @onnx9000.jit
    def model(a):
        """Provides model functionality and verification."""
        return onnx9000.core.ops.depth_to_space(a, blocksize=2, mode="DCR")

    a = onnx9000.Tensor(shape=(1, 4, 2, 2), dtype=DType.FLOAT32, name="a")
    builder = model(a)
    out_path = temp_dir / "depthtospace.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.arange(16, dtype=np.float32).reshape((1, 4, 2, 2))
    output = compiled(a_val)
    res = output[0].reshape((1, 1, 4, 4))
    assert res.shape == (1, 1, 4, 4)


def test_spacetodepth(temp_dir: Path):
    """Tests the test spacetodepth functionality."""

    @onnx9000.jit
    def model2(a):
        """Provides model2 functionality and verification."""
        return onnx9000.core.ops.space_to_depth(a, blocksize=2)

    a = onnx9000.Tensor(shape=(1, 1, 4, 4), dtype=DType.FLOAT32, name="a")
    builder = model2(a)
    out_path = temp_dir / "spacetodepth.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.arange(16, dtype=np.float32).reshape((1, 1, 4, 4))
    output = compiled(a_val)
    res = output[0].reshape((1, 4, 2, 2))
    assert res.shape == (1, 4, 2, 2)
