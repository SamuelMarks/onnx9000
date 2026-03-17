"""Test operations."""

from onnx9000.core.dtypes import DType
from onnx9000.converters.frontend.builder import GraphBuilder, Tracing
from onnx9000.converters.frontend.tensor import Tensor


def test_tensor_ops() -> None:
    """Tests the corresponding tensor functionality."""
    builder = GraphBuilder("test_ops")
    t1 = Tensor((2, 2), DType.FLOAT32, "t1")
    t2 = Tensor((2, 2), DType.FLOAT32, "t2")
    with Tracing(builder):
        t3 = t1 + t2
        t1 - t2
        t1 * t2
        t1 / t2
        t1 @ t2
        t1**t2
        t1 % t2
        1.0 + t1
        1.0 - t1
        1.0 * t1
        1.0 / t1
        abs(t1)
        t1[0]
        t1[1] = t2
        t1.sum(0)
        t1.mean(1, keepdim=True)
        t1.max()
        t1.min((0, 1))
        t1.transpose(0, 1)
        t1.reshape((4,))
        t1.view(4)
        t1.squeeze()
        t1.unsqueeze(0)
        t1.flatten()
        t1.expand(4, 2, 2)
        t1.broadcast_to((4, 2, 2))
        t1.contiguous()
        t1.type(DType.INT32)
        t1.to(DType.FLOAT64)
        t1.exp()
        t1.log()
        t1.sqrt()
        t1.sin()
        t1.cos()
        t1.tan()
        t1.asin()
        t1.acos()
        t1.atan()
        t1.sinh()
        t1.cosh()
        t1.relu()
        t1.sigmoid()
        t1.tanh()
        t1.gelu()
        t1.softmax()
        t49 = t1 == t2
        t1 & t2
        t1 | t2
        t1 ^ t2
        t1.where(t49, t2)
        t1.clip(0.0, 1.0)
        t1.clamp(min_val=0.0)
        t1.argmax(0)
        t1.argmin(1)
        t1.gather(0, t2)
        t1.scatter(0, t2, t3)
        t1.masked_select(t49)
        t1.nonzero()
        t1.clone()
        t1.detach()
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
