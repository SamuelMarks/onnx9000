"""Test operations."""

import pytest
from onnx9000 import Tensor, GraphBuilder, Tracing
from onnx9000.core.dtypes import DType


def test_tensor_ops():
    """Provides semantic functionality and verification."""
    builder = GraphBuilder("test_ops")
    t1 = Tensor((2, 2), DType.FLOAT32, "t1")
    t2 = Tensor((2, 2), DType.FLOAT32, "t2")
    with Tracing(builder):
        t3 = t1 + t2
        t4 = t1 - t2
        t5 = t1 * t2
        t6 = t1 / t2
        t7 = t1 @ t2
        t8 = t1**t2
        t9 = t1 % t2
        t10 = 1.0 + t1
        t11 = 1.0 - t1
        t12 = 1.0 * t1
        t13 = 1.0 / t1
        t14 = -t1
        t15 = abs(t1)
        t16 = t1[0]
        t1[1] = t2
        t17 = t1.sum(0)
        t18 = t1.mean(1, keepdim=True)
        t19 = t1.max()
        t20 = t1.min((0, 1))
        t21 = t1.transpose(0, 1)
        t22 = t1.T
        t23 = t1.reshape((4,))
        t24 = t1.view(4)
        t25 = t1.squeeze()
        t26 = t1.unsqueeze(0)
        t27 = t1.flatten()
        t28 = t1.expand(4, 2, 2)
        t29 = t1.broadcast_to((4, 2, 2))
        t30 = t1.contiguous()
        t31 = t1.type(DType.INT32)
        t32 = t1.to(DType.FLOAT64)
        t33 = t1.exp()
        t34 = t1.log()
        t35 = t1.sqrt()
        t36 = t1.sin()
        t37 = t1.cos()
        t38 = t1.tan()
        t39 = t1.asin()
        t40 = t1.acos()
        t41 = t1.atan()
        t42 = t1.sinh()
        t43 = t1.cosh()
        t44 = t1.relu()
        t45 = t1.sigmoid()
        t46 = t1.tanh()
        t47 = t1.gelu()
        t48 = t1.softmax()
        t49 = t1 == t2
        t50 = t1 != t2
        t51 = t1 < t2
        t52 = t1 <= t2
        t53 = t1 > t2
        t54 = t1 >= t2
        t55 = t1 & t2
        t56 = t1 | t2
        t57 = t1 ^ t2
        t58 = ~t1
        t59 = t1.where(t49, t2)
        t60 = t1.clip(0.0, 1.0)
        t61 = t1.clamp(min_val=0.0)
        t62 = t1.argmax(0)
        t63 = t1.argmin(1)
        t64 = t1.gather(0, t2)
        t65 = t1.scatter(0, t2, t3)
        t66 = t1.masked_select(t49)
        t67 = t1.nonzero()
        t68 = t1.clone()
        t69 = t1.detach()
    assert builder.nodes
    t_data = Tensor((1,), DType.FLOAT32, data=42.0)
    assert t_data.item() == 42.0
    assert t_data.tolist() == [42.0] or t_data.tolist() == 42.0
    assert t_data.numpy() == 42.0
    t_no_data = Tensor((1,), DType.FLOAT32)
    assert t_no_data.item() is None
    assert t_no_data.tolist() is None
    assert t_no_data.numpy() is None
    t_no_data.grad = t_data
    assert t_no_data.grad is t_data
    repr(t1)
    with Tracing(builder):
        t1.sum((0, 1))
        t1.squeeze((0,))
        t1.view([4])
        t1.clip(max_val=1.0)
