"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Parameter, Tensor
from onnx9000.core.dtypes import DType


def test_module_init_and_attr() -> None:
    """Tests the corresponding module functionality."""

    class M(Module):
        """A mock module for testing parameter and sub-module registration."""

        def __init__(self) -> None:
            """Initialize the mock module with parameters, buffers, and sub-modules."""
            self.initialized = False
            super().__init__()
            self.p = Parameter((10,), DType.FLOAT32, "p")
            self.b = Tensor((10,), DType.FLOAT32, "b")
            self.m = Module()
            self.val = 5

    m = M()
    assert hasattr(m, "p")
    assert isinstance(m.p, Parameter)
    assert hasattr(m, "b")
    assert isinstance(m.b, Tensor)
    assert hasattr(m, "m")
    assert isinstance(m.m, Module)
    assert m.val == 5
    m.p = Module()
    assert isinstance(m.p, Module)
    m.b = Parameter((5,), DType.FLOAT32, "new_p")
    assert isinstance(m.b, Parameter)
    m.m = Tensor((5,), DType.FLOAT32, "new_b")
    assert isinstance(m.m, Tensor)
    m.m = 10
    assert m.m == 10
    with pytest.raises(AttributeError):
        m.missing_attr


def test_module_errors() -> None:
    """Tests the corresponding module functionality."""

    class NoInit(Module):
        """A mock module that fails to call super().__init__()."""

        def __init__(self) -> None:
            """Initialize the mock module with parameters, buffers, and sub-modules."""
            self.initialized = False

        __dummy__ = True

    n = NoInit()
    with pytest.raises(AttributeError):
        n.p = Parameter((1,), DType.FLOAT32, "p")
    m = Module()
    with pytest.raises(TypeError):
        m.register_parameter(123, None)
    with pytest.raises(TypeError):
        m.register_parameter("p", Tensor((1,), DType.FLOAT32))
    with pytest.raises(TypeError):
        m.register_buffer(123, None)
    with pytest.raises(TypeError):
        m.add_module(123, None)
    with pytest.raises(TypeError):
        m.add_module("m", "not a module")


def test_module_iterators() -> None:
    """Tests the corresponding module functionality."""
    m = Module()
    m.p = Parameter((1,), DType.FLOAT32, "p")
    m.b = Tensor((1,), DType.FLOAT32, "b")
    m.m = Module()
    m.m.p2 = Parameter((1,), DType.FLOAT32, "p2")
    m.none_p = None
    m.register_parameter("none_p", None)
    m.register_buffer("none_b", None)
    m.add_module("none_m", None)
    assert len(list(m.parameters())) == 2
    assert len(list(m.named_parameters())) == 2
    assert len(list(m.buffers())) == 1
    assert len(list(m.named_buffers())) == 1
    assert len(list(m.children())) == 1
    assert len(list(m.named_children())) == 1
    assert len(list(m.modules())) == 2


def test_module_state_dict() -> None:
    """Tests the corresponding module functionality."""
    m = Module()
    m.p = Parameter((1,), DType.FLOAT32, "p", data=np.array([1.0], dtype=np.float32))
    m.b = Tensor((1,), DType.FLOAT32, "b", data=np.array([2.0], dtype=np.float32))
    m.m = Module()
    m.m.p2 = Parameter((1,), DType.FLOAT32, "p2", data=np.array([3.0], dtype=np.float32))
    m.register_parameter("none_p", None)
    m.register_buffer("none_b", None)
    m.add_module("none_m", None)
    sd = m.state_dict()
    assert "p" in sd
    assert "b" in sd
    assert "m.p2" in sd
    m2 = Module()
    m2.p = Parameter((1,), DType.FLOAT32, "p")
    m2.b = Tensor((1,), DType.FLOAT32, "b")
    m2.m = Module()
    m2.m.p2 = Parameter((1,), DType.FLOAT32, "p2")
    m2.load_state_dict(sd)
    assert np.array_equal(m2.p.data, np.array([1.0], dtype=np.float32))


def test_module_load_state_dict_errors() -> None:
    """Tests the corresponding module functionality."""
    m = Module()
    m.p = Parameter((1,), DType.FLOAT32, "p")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        m.load_state_dict({"bad": 1.0})
    with pytest.raises(RuntimeError, match="Missing key"):
        m.load_state_dict({})


def test_module_methods() -> None:
    """Tests the corresponding module functionality."""
    m = Module()
    m.p = Parameter((1,), DType.FLOAT32, "p")
    m.b = Tensor((1,), DType.FLOAT32, "b")
    m.to("cpu")
    m.eval()
    assert not m.training
    m.train()
    assert m.training

    def fn(mod) -> None:
        """A mock visitor function to test module.apply()."""
        mod.visited = True

    m.apply(fn)
    assert m.visited
    m.p.grad = Tensor((1,), DType.FLOAT32, "g")
    m.zero_grad()
    assert m.p.grad is None
    m.register_parameter("none_p", None)
    m.zero_grad()


def test_module_forward() -> None:
    """Tests the corresponding module functionality."""
    m = Module()
    assert m(1) is None


def test_setattr_coverage() -> None:
    """Tests the corresponding module functionality."""
    m = Module()
    m._parameters["test"] = Parameter((1,), DType.FLOAT32, "test")
    m.test = Module()
    m = Module()
    m._parameters["test2"] = Parameter((1,), DType.FLOAT32, "test2")
    m.test2 = 5


def test_module_recurse_train_apply() -> None:
    """Tests the corresponding module functionality."""
    m = Module()
    sub = Module()
    m.add_module("sub", sub)
    m.train()
    assert sub.training
    m.eval()
    assert not sub.training

    def fn(mod) -> None:
        """A mock visitor function to test module.apply()."""
        mod.visited_recurse = True

    m.apply(fn)
    assert sub.visited_recurse


def test_module_setattr_edge_cases() -> None:
    """Tests edge cases in __setattr__ where previous assignments are overridden."""
    from onnx9000.converters.frontend.nn.module import Module, Parameter
    from onnx9000.converters.frontend.tensor import Tensor

    m = Module()
    m.add_module("p1", Module())
    m.p1 = Parameter(Tensor([1, 2, 3]))
    m.__dict__["m1"] = "some value"
    m.m1 = Module()
    m.register_buffer("m2", Tensor([1]))
    m.m2 = Module()
    m.__dict__["t1"] = "another value"
    m.t1 = Tensor([4])
    m.register_parameter("t2", Parameter(Tensor([5])))
    m.t2 = Tensor([6])
    m.add_module("r1", Module())
    m.r1 = "string object"


def test_module_setattr_dict_override() -> None:
    """Tests the test module setattr dict override functionality."""
    from onnx9000.converters.frontend.nn.module import Module, Parameter
    from onnx9000.converters.frontend.tensor import Tensor

    m = Module()
    m.__dict__["p2"] = "some value"
    m.p2 = Parameter(Tensor([1, 2, 3]))


def test_module_register_buffer_errors() -> None:
    """Tests that register_buffer throws TypeError on invalid tensor argument."""
    import pytest
    from onnx9000.converters.frontend.nn.module import Module

    m = Module()
    with pytest.raises(TypeError, match="cannot assign to buffer, must be a Tensor or None"):
        m.register_buffer("bad", 123)
