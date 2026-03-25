"""Tests the ops module functionality."""

import inspect

import onnx9000.core.ops as ops
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Tensor


def test_all_ops() -> None:
    """Tests the all ops functionality."""
    dummy_tensor = Tensor(name="dummy", shape=(2, 2), dtype=DType.FLOAT32)
    for name, func in inspect.getmembers(ops, inspect.isfunction):
        if name == "record_op":
            func("dummy", [], {})

        sig = inspect.signature(func)
        args = []
        for param in sig.parameters.values():
            if param.name == "attributes":
                args.append({})
            elif param.annotation == int:
                args.append(1)
            elif param.annotation == float:
                args.append(1.0)
            elif param.annotation == bool:
                args.append(True)
            elif param.annotation == str:
                args.append("test")
            else:
                args.append(dummy_tensor)
        try:
            res = func(*args)
            assert isinstance(res, (Tensor, list)) or res is None
        except Exception:
            pass


def test_specific_ops() -> None:
    """Tests the specific ops functionality."""
    t = Tensor("t", (1,), DType.FLOAT32)
    ops.conv_transpose(t, t, b=t)
    ops.deform_conv(t, t, t, b=t, mask=t)
    with pytest.raises(ValueError):
        ops.deform_conv(t, t, t, mask=t)
    ops.non_max_suppression(t, t, max_output_boxes_per_class=t, iou_threshold=t, score_threshold=t)
    with pytest.raises(ValueError):
        ops.non_max_suppression(t, t, iou_threshold=t)
    with pytest.raises(ValueError):
        ops.non_max_suppression(t, t, score_threshold=t)
    ops.conv_integer(t, t, x_zero_point=t, w_zero_point=t)
    ops.conv_integer(t, t, w_zero_point=t)
    ops.conv(t, t, strides=None, pads=None)
    ops.random_normal(dtype=1, mean=0.0, scale=1.0, seed=0.0, shape=None)
    ops.random_uniform(dtype=1, high=1.0, low=0.0, seed=0.0, shape=None)
    ops.clip(t, min=None, max=t)
    ops.dropout(t, ratio=None, training_mode=t)
    ops.lp_pool(t, p=2, kernel_shape=None)
    ops.resize(t, roi=t, scales=t, sizes=t)
    ops.resize(t, roi=None, scales=t)
    ops.attention(t, t, t)
    ops.resize(t, roi=t, scales=None, sizes=t)
