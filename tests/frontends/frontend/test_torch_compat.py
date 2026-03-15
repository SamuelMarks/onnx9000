"""Module providing core logic and structural definitions."""

import onnx9000.frontends.torch_like as torch
import pytest


def test_drop_in_replacement():
    """Provides semantic functionality and verification."""

    class MyModel(torch.nn.Module):
        """Provides semantic functionality and verification."""

        def __init__(self):
            """Provides semantic functionality and verification."""
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)

        def forward(self, x):
            """Provides semantic functionality and verification."""
            return torch.nn.functional.relu(self.linear(x))

    m = MyModel()
    x = torch.randn(5, 10, dtype=torch.float32)
    traced_model = torch.jit.trace(m, x)
    assert len(traced_model.nodes) > 0
    import io

    buffer = io.BytesIO()
    torch.onnx.export(m, x, buffer)
    assert len(buffer.getvalue()) > 0
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.zeros(2, 2)
    t3 = torch.ones(2, 2)
    assert t1.shape == (2,)
    assert t2.shape == (2, 2)
    assert t3.shape == (2, 2)


def test_torch_like_missing():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.torch_like import tensor, jit
    from onnx9000.core.dtypes import DType

    t1 = tensor([1, 2], dtype=DType.INT32)
    assert t1.dtype == DType.INT32

    def my_fn(a):
        """Provides my fn functionality and verification."""
        return a

    jit.trace(my_fn, t1)
    jit.script(my_fn, t1)


def test_trace_non_tensor_and_kwargs():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.torch_like import tensor, jit
    from onnx9000.core.dtypes import DType

    t1 = tensor([1, 2], dtype=DType.INT32)

    def my_fn2(a, b, c=None):
        """Provides my fn2 functionality and verification."""
        return a

    jit.trace(my_fn2, t1, 5, c=t1)
