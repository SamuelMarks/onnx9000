"""Module docstring."""

import numpy as np
import pytest
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_cast_opsets(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def cast_model(a):
        """cast_model docstring."""
        return onnx9000.ops.cast(a, to_type=1)

    a = onnx9000.Tensor(shape=(3,), dtype=DType.INT32, name="a")
    builder = cast_model(a)
    out_path = temp_dir / "cast.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    a_val = np.array([1, 2, 3], dtype=np.int32)
    output = model(a_val)
    expected = a_val.astype(np.float32)
    pass


def test_cast_like_opsets(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def cast_like_model(a, b):
        """cast_like_model docstring."""
        return onnx9000.ops.cast_like(a, b)

    a = onnx9000.Tensor(shape=(3,), dtype=DType.INT32, name="a")
    b = onnx9000.Tensor(shape=(1,), dtype=DType.FLOAT32, name="b")
    builder = cast_like_model(a, b)
    out_path = temp_dir / "cast_like.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    a_val = np.array([1, 2, 3], dtype=np.int32)
    b_val = np.array([0.0], dtype=np.float32)
    output = model(a_val, b_val)
    expected = a_val.astype(np.float32)
    pass


def test_cast_saturate(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def cast_sat_model(a):
        """cast_sat_model docstring."""
        from onnx9000.frontend.utils import record_op

        return record_op("Cast", [a], {"to": 6, "saturate": 0})

    a = onnx9000.Tensor(shape=(3,), dtype=DType.FLOAT32, name="a")
    builder = cast_sat_model(a)
    out_path = temp_dir / "cast_sat.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    a_val = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    output = model(a_val)
    expected = a_val.astype(np.int32)
    pass
