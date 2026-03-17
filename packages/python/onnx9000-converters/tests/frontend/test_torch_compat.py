"""Module providing core logic and structural definitions."""

import onnx9000.converters.torch_like as torch


def test_drop_in_replacement() -> None:
    """Tests the test_drop_in_replacement functionality."""

    class MyModel(torch.nn.Module):
        """Class MyModel implementation."""

        def __init__(self) -> None:
            """Tests the __init__ functionality."""
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)

        def forward(self, x):
            """Tests the forward functionality."""
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


def test_torch_like_missing() -> None:
    """Tests the test_torch_like_missing functionality."""
    from onnx9000.core.dtypes import DType
    from onnx9000.converters.torch_like import jit, tensor

    t1 = tensor([1, 2], dtype=DType.INT32)
    assert t1.dtype == DType.INT32

    def my_fn(a):
        """Tests the my_fn functionality."""
        return a

    jit.trace(my_fn, t1)
    jit.script(my_fn, t1)


def test_trace_non_tensor_and_kwargs() -> None:
    """Tests the test_trace_non_tensor_and_kwargs functionality."""
    from onnx9000.core.dtypes import DType
    from onnx9000.converters.torch_like import jit, tensor

    t1 = tensor([1, 2], dtype=DType.INT32)

    def my_fn2(a, b, c=None):
        """Tests the my_fn2 functionality."""
        return a

    jit.trace(my_fn2, t1, 5, c=t1)
