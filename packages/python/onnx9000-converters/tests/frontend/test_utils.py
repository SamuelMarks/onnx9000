"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.converters.frontend.builder import GraphBuilder, Tracing
from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.converters.frontend.utils import (
    infer_elementwise_shape,
    infer_matmul_shape,
    record_op,
)


def test_infer_elementwise_shape() -> None:
    """Tests the test_infer_elementwise_shape functionality."""
    assert infer_elementwise_shape((10,), (10,)) == (10,)
    assert infer_elementwise_shape((1, 10), (10,)) == (1, 10)
    assert infer_elementwise_shape((5, 1, 10), (1, 3, 10)) == (5, 3, 10)
    assert infer_elementwise_shape(("batch",), (1,)) == ("batch",)
    assert infer_elementwise_shape((1,), ("batch",)) == ("batch",)
    with pytest.raises(ValueError):
        infer_elementwise_shape((5,), (10,))


def test_infer_matmul_shape() -> None:
    """Tests the test_infer_matmul_shape functionality."""
    assert infer_matmul_shape((10,), (10,)) == ()
    assert infer_matmul_shape((10,), (10, 5)) == (5,)
    assert infer_matmul_shape((5, 10), (10,)) == (5,)
    assert infer_matmul_shape((2, 5, 10), (10, 3)) == (2, 5, 3)
    with pytest.raises(ValueError):
        infer_matmul_shape((), (10,))


def test_record_op_no_context() -> None:
    """Tests the test_record_op_no_context functionality."""
    with pytest.raises(RuntimeError):
        record_op("Add", [Tensor()])


def test_record_op_numpy_constant() -> None:
    """Tests the test_record_op_numpy_constant functionality."""
    gb = GraphBuilder("test")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        res = record_op("Add", [t1, np.array([1.0], dtype=np.float32)])
        assert res.shape == (10,)
        record_op("Add", [t1, np.array([1], dtype=np.int32)])
        record_op("Add", [t1, np.array([1], dtype=np.int64)])
        record_op("Add", [t1, np.array([1.0], dtype=np.float64)])
        record_op("Add", [t1, np.array([True], dtype=bool)])
        record_op("Add", [t1, [1.0]])


def test_record_op_special_ops() -> None:
    """Tests the test_record_op_special_ops functionality."""
    gb = GraphBuilder("test")
    with Tracing(gb):
        t = Tensor((10,), DType.FLOAT32)
        res = record_op("TopK", [t])
        assert len(res) == 2
        res = record_op("DynamicQuantizeLinear", [t])
        assert len(res) == 3
        res = record_op("Dropout", [t])
        assert len(res) == 2
        res = record_op("Unique", [t])
        assert len(res) == 4
        res = record_op("Transpose", [t])
        assert res.shape == (10,)
        res = record_op("Trilu", [t])
        assert res.dtype == DType.FLOAT32


def test_tensor_init_dtypes() -> None:
    """Tests the test_tensor_init_dtypes functionality."""
    t1 = Tensor(data=np.array([1], dtype=np.int64))
    assert t1.dtype == DType.INT64
    t2 = Tensor(data=np.array([1], dtype=np.int32))
    assert t2.dtype == DType.INT32
    t3 = Tensor(data=np.array([1.0], dtype=np.float64))
    assert t3.dtype == DType.FLOAT64
    t4 = Tensor(data=np.array([True], dtype=bool))
    assert t4.dtype == DType.BOOL
    t5 = Tensor(data=np.array([1.0], dtype=np.float32))
    assert t5.dtype == DType.FLOAT32


def test_tensor_init_bad_data() -> None:
    """Tests the test_tensor_init_bad_data functionality."""

    class Bad:
        """Class Bad implementation."""

        def __array__(self):
            """Tests the __array__ functionality."""
            raise ValueError("bad")

    t = Tensor(data=Bad())
    assert t.data is None


def test_tensor_ops() -> None:
    """Tests the test_tensor_ops functionality."""
    gb = GraphBuilder("test")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        t2 = Tensor((10,), DType.FLOAT32)
        _ = t1 + t2
        _ = 1.0 + t1
        _ = t1 - t2
        _ = 1.0 - t1
        _ = t1 * t2
        _ = 1.0 * t1
        _ = t1 / t2
        _ = 1.0 / t1
        _ = t1 @ t2
        _ = t1**2
        _ = t1 % 2
        _ = -t1
        _ = abs(t1)
        _ = t1[0]
        t1[0] = 5.0
        with pytest.raises(RuntimeError):
            bool(t1)
        _ = t1 & t2
        _ = t1 | t2
        _ = t1 ^ t2
        _ = ~t1
        _ = t1 == t2
        _ = t1 != t2
        _ = t1 < t2
        _ = t1 <= t2
        _ = t1 > t2
        _ = t1 >= t2
        _ = t1.sum()
        _ = t1.mean()
        _ = t1.max()
        _ = t1.min()
        _ = t1.transpose(0, 0)
        t_scalar = Tensor(())
        _ = t_scalar.transpose(0, 0)
        _ = t1.reshape((2, 5))
        _ = t1.view(2, 5)
        _ = t1.squeeze()
        _ = t1.unsqueeze(0)
        _ = t1.flatten()
        _ = t1.expand(1, 10)
        _ = t1.broadcast_to((1, 10))
        _ = t1.contiguous()
        _ = t1.type(DType.INT32)
        _ = t1.to("cpu")
        _ = t1.exp()
        _ = t1.log()
        _ = t1.sqrt()
        _ = t1.sin()
        _ = t1.cos()
        _ = t1.tan()
        _ = t1.asin()
        _ = t1.acos()
        _ = t1.atan()
        _ = t1.sinh()
        _ = t1.cosh()
        _ = t1.relu()
        _ = t1.sigmoid()
        _ = t1.tanh()
        _ = t1.gelu()
        _ = t1.softmax()
        _ = t1.where(t2, t1)
        _ = t1.clip(0, 1)
        _ = t1.clamp(0, 1)
        _ = t1.argmax()
        _ = t1.argmin()
        _ = t1.gather(0, t2)
        _ = t1.scatter(0, t2, t1)
        _ = t1.masked_select(t2)
        _ = t1.nonzero()
        t1.requires_grad = True
        assert t1.requires_grad
        t1.grad = t2
        assert t1.grad is t2
        _ = t1.detach()
        _ = t1.clone()


def test_tensor_item_tolist_numpy() -> None:
    """Tests the test_tensor_item_tolist_numpy functionality."""
    t1 = Tensor((1,), DType.FLOAT32, data=np.array([1.0], dtype=np.float32))
    assert t1.item() == 1.0
    assert t1.tolist() == [1.0]
    assert np.array_equal(t1.numpy(), np.array([1.0], dtype=np.float32))
    t2 = Tensor((1,), DType.FLOAT32)
    assert t2.item() is None
    assert t2.tolist() is None
    assert t2.numpy() is None
    t1.requires_grad_(True)


def test_record_op_transpose_missing() -> None:
    """Tests the test_record_op_transpose_missing functionality."""
    gb = GraphBuilder("test")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        record_op("Transpose", [t1])


def test_tensor_transpose_prop() -> None:
    """Tests the test_tensor_transpose_prop functionality."""
    gb = GraphBuilder("test")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = t1.T


def test_infer_elementwise_dynamic() -> None:
    """Tests the test_infer_elementwise_dynamic functionality."""
    assert infer_elementwise_shape(("batch",), (1,)) == ("batch",)
    assert infer_elementwise_shape((1,), ("batch",)) == ("batch",)
    assert infer_elementwise_shape(("batch",), ("batch",)) == ("batch",)


def test_infer_elementwise_dynamic_missing() -> None:
    """Tests the test_infer_elementwise_dynamic_missing functionality."""
    assert infer_elementwise_shape((1,), ("batch",)) == ("batch",)
    assert infer_elementwise_shape(("batch",), (2,)) == ("batch",)


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_constructors() -> None:
    """Tests the test_tensor_constructors functionality."""
    t1 = Tensor(data=np.array([1], dtype=np.int64))
    assert t1.dtype == DType.INT64
    t2 = Tensor(data=np.array([1], dtype=np.int32))
    assert t2.dtype == DType.INT32
    t3 = Tensor(data=np.array([1.0], dtype=np.float64))
    assert t3.dtype == DType.FLOAT64
    t4 = Tensor(data=np.array([True], dtype=bool))
    assert t4.dtype == DType.BOOL
    t5 = Tensor(data=np.array([1], dtype=np.uint8))
    assert t5.dtype == DType.FLOAT32


def test_tensor_ops_remaining() -> None:
    """Tests the test_tensor_ops_remaining functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = t1.__rsub__(1.0)
        _ = t1.__rtruediv__(1.0)
        t1.requires_grad = False
        t1.requires_grad = True
        t_scalar = Tensor(())
        _ = t_scalar.transpose(0, 0)
        _ = t1.transpose(-1, -1)


def test_tensor_setitem() -> None:
    """Tests the test_tensor_setitem functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        t1[0] = 5.0


def test_tracer_exceptions_and_args() -> None:
    """Tests the test_tracer_exceptions_and_args functionality."""
    from onnx9000.converters.frontend.tracer import trace

    def my_fn(a, b, k=1):
        """Tests the my_fn functionality."""
        if k == 1:
            raise ValueError("bad")
        return a

    t1 = Tensor((1,), DType.FLOAT32, "a")
    with pytest.raises(RuntimeError, match="Tracing failed: bad"):
        trace(my_fn, t1, 5)


def test_tracer_returns() -> None:
    """Tests the test_tracer_returns functionality."""
    from onnx9000.converters.frontend.tracer import trace

    def my_fn_tup(a):
        """Tests the my_fn_tup functionality."""
        return (a,)

    def my_fn_list(a):
        """Tests the my_fn_list functionality."""
        return [a]

    def my_fn_dict(a):
        """Tests the my_fn_dict functionality."""
        return {"a": a}

    t1 = Tensor((1,), DType.FLOAT32, "a")
    trace(my_fn_tup, t1)
    trace(my_fn_list, t1)
    trace(my_fn_dict, t1)


def test_tracer_script() -> None:
    """Tests the test_tracer_script functionality."""
    from onnx9000.converters.frontend.tracer import script

    def my_fn(a):
        """Tests the my_fn functionality."""
        return a

    t = Tensor((1,), DType.FLOAT32, "a")
    script(my_fn, t)


def test_tracer_returns_dict() -> None:
    """Tests the test_tracer_returns_dict functionality."""
    from onnx9000.converters.frontend.tracer import trace

    def my_fn_dict(a):
        """Tests the my_fn_dict functionality."""
        return {"a": a}

    t1 = Tensor((1,), DType.FLOAT32, "a")
    trace(my_fn_dict, t1)


def test_tracer_proxy_kwargs_else() -> None:
    """Tests the test_tracer_proxy_kwargs_else functionality."""
    from onnx9000.converters.frontend.tracer import trace

    def my_fn(a, k=1):
        """Tests the my_fn functionality."""
        return a

    t1 = Tensor((1,), DType.FLOAT32, "a")
    trace(my_fn, t1, k="not_a_tensor")


def test_tracer_proxy_kwargs_if() -> None:
    """Tests the test_tracer_proxy_kwargs_if functionality."""
    from onnx9000.converters.frontend.tracer import trace

    def my_fn(a, k=None):
        """Tests the my_fn functionality."""
        return a + k

    t1 = Tensor((1,), DType.FLOAT32, "a")
    t2 = Tensor((1,), DType.FLOAT32, "k")
    trace(my_fn, t1, k=t2)


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test_tensor_r_ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test_tensor_transpose_neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test_tensor_grad_prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test tensor r ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test tensor transpose neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test tensor grad prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    """Tests the test tensor r ops functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    """Tests the test tensor transpose neg functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    """Tests the test tensor grad prop functionality."""
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1


def test_tensor_r_ops() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10,), DType.FLOAT32)
        _ = 1.0 - t1
        _ = 1.0 / t1


def test_tensor_transpose_neg() -> None:
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t1 = Tensor((10, 5), DType.FLOAT32)
        _ = t1.transpose(-1, -2)


def test_tensor_grad_prop() -> None:
    t1 = Tensor((10,), DType.FLOAT32)
    assert t1.grad is None
    t1.grad = t1
    assert t1.grad is t1
