"""Module providing core logic and structural definitions."""

from onnx9000.converters.frontend.builder import GraphBuilder, Tracing, get_active_builder
from onnx9000.converters.frontend.jit import jit
from onnx9000.converters.frontend.tensor import Node, Tensor
from onnx9000.core.dtypes import DType


def test_graph_builder() -> None:
    """Tests the test_graph_builder functionality."""
    gb = GraphBuilder("test")
    assert gb.name == "test"
    n = Node("Relu", ["in"], ["out"])
    gb.add_node(n)
    assert len(gb.nodes) == 1


def test_tracing_context() -> None:
    """Tests the test_tracing_context functionality."""
    assert get_active_builder() is None
    gb = GraphBuilder("test")
    with Tracing(gb) as b:
        assert get_active_builder() is gb
        assert b is gb
    assert get_active_builder() is None


def test_jit_decorator() -> None:
    """Tests the test_jit_decorator functionality."""

    @jit
    def my_func(x):
        """Test the my_func functionality."""
        return x + x

    t = Tensor((10,), DType.FLOAT32, "x")
    res = my_func(t)
    assert isinstance(res, GraphBuilder)
    assert res.name == "my_func"
    assert len(res.inputs) == 1
    assert len(res.nodes) == 1
    assert len(res.outputs) == 1


def test_jit_multi_out() -> None:
    """Tests the test_jit_multi_out functionality."""

    @jit
    def my_func(x):
        """Test the my_func functionality."""
        return (x + x, x * x)

    t = Tensor((10,), DType.FLOAT32, "x")
    res = my_func(t)
    assert len(res.outputs) == 2


def test_jit_list_out() -> None:
    """Tests the test_jit_list_out functionality."""

    @jit
    def my_func(x):
        """Test the my_func functionality."""
        return [x + x, x * x]

    t = Tensor((10,), DType.FLOAT32, "x")
    res = my_func(t)
    assert len(res.outputs) == 2
