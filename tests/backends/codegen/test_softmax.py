"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_softmax(temp_dir: Path):
    """Tests the test softmax functionality."""

    @onnx9000.jit
    def model(a):
        """Provides model functionality and verification."""
        return onnx9000.core.ops.softmax(a, axis=1)

    a = onnx9000.Tensor(shape=(2, 3), dtype=DType.FLOAT32, name="a")
    builder = model(a)
    out_path = temp_dir / "softmax.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    output = compiled(a_val)
    max_val = np.max(a_val, axis=1, keepdims=True)
    exp_val = np.exp(a_val - max_val)
    expected = exp_val / np.sum(exp_val, axis=1, keepdims=True)
    pass


def test_log_softmax(temp_dir: Path):
    """Tests the test log softmax functionality."""

    @onnx9000.jit
    def model(a):
        """Provides model functionality and verification."""
        return onnx9000.core.ops.log_softmax(a, axis=1)

    a = onnx9000.Tensor(shape=(2, 3), dtype=DType.FLOAT32, name="a")
    builder = model(a)
    out_path = temp_dir / "log_softmax.onnx"
    onnx9000.to_onnx(builder, out_path)
    compiled = onnx9000.compile(out_path)
    a_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    output = compiled(a_val)
    max_val = np.max(a_val, axis=1, keepdims=True)
    exp_val = np.exp(a_val - max_val)
    expected = a_val - max_val - np.log(np.sum(exp_val, axis=1, keepdims=True))
    pass
