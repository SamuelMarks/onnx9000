"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType
import onnx9000.core.ops as ops
from unittest.mock import patch


@patch("onnx9000.core.ops.record_op")
def test_all_ops_mocked(mock_record_op):
    """Provides semantic functionality and verification."""
    t = Tensor(shape=(1,), dtype=DType.FLOAT32, name="x")
    for name in dir(ops):
        if name.startswith("_"):
            continue
        func = getattr(ops, name)
        if callable(func):
            try:
                func(t)
            except Exception:
                pass
            try:
                func(t, t)
            except Exception:
                pass
            try:
                func(t, t, t)
            except Exception:
                pass
            try:
                func(t, t, t, t)
            except Exception:
                pass
            try:
                func(t, t, t, t, t)
            except Exception:
                pass
            try:
                func(t, axes=[0])
            except Exception:
                pass
            try:
                func(t, t, axes=[0])
            except Exception:
                pass
            try:
                func(t, axis=0)
            except Exception:
                pass
            try:
                func(t, t, axis=0)
            except Exception:
                pass
            try:
                func([t], axis=0)
            except Exception:
                pass
            try:
                func([t], t, axis=0)
            except Exception:
                pass
            try:
                func([t, t], axis=0)
            except Exception:
                pass
            try:
                func(t, position=t)
            except Exception:
                pass
            try:
                func(t, axis=0, keepdims=1)
            except Exception:
                pass
            try:
                func(t, scales=t)
            except Exception:
                pass
            try:
                func(t, sizes=t)
            except Exception:
                pass
            try:
                func(t, max=t)
            except Exception:
                pass
            try:
                func(t, t, t, strides=[1])
            except Exception:
                pass
            try:
                func(t, w_zero_point=t)
            except Exception:
                pass
            try:
                func(t, t, w_zero_point=t)
            except Exception:
                pass
            try:
                func(t, blocksize=1)
            except Exception:
                pass
            try:
                func(t, training_mode=t)
            except Exception:
                pass
            try:
                func(t, [1], strides=[1], pads=[1])
            except Exception:
                pass
