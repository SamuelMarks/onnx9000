"""Tests the coverage gap module functionality."""

import os
import tempfile

from onnx9000.core.dtypes import DType
from onnx9000.core.execution import ExecutionContext, ExecutionProvider, SessionOptions
from onnx9000.core.ir import Attribute, Constant, DynamicDim, Graph, Node, Tensor
from onnx9000.core.memory import mmap_tensor_data
from onnx9000.core.serializer import save, to_bytes
from onnx9000.core.symbolic import broadcast_shapes, evaluate_symbolic_expression


def test_mmap_tensor_data() -> None:
    """Tests the mmap tensor data functionality."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"hello world")
        name = f.name
    view = mmap_tensor_data(name, 0, 5)
    assert view.tobytes() == b"hello"
    view = mmap_tensor_data(name, 6, 5)
    assert view.tobytes() == b"world"
    view = mmap_tensor_data(name, 6, 0)
    assert view.tobytes() == b""
    os.remove(name)


def test_symbolic_math() -> None:
    """Tests the symbolic math functionality."""
    res = evaluate_symbolic_expression("batch", {"batch": 8})
    assert res == 8
    res = evaluate_symbolic_expression("seq", {"batch": 8})
    assert res == "seq"
    shape1 = (1, DynamicDim("seq"), 20)
    shape2 = (8, 1, 20)
    res = broadcast_shapes(shape1, shape2)
    assert res[0] == 8
    assert isinstance(res[1], DynamicDim)
    assert res[1].value == "seq"
    assert res[2] == 20


def test_execution_provider_base() -> None:
    """Tests the execution provider base functionality."""

    class DummyEP(ExecutionProvider):
        """Represents the DummyEP class and its associated logic."""

        def get_supported_nodes(self, graph):
            """Tests the get supported nodes functionality."""
            return []

        def allocate_tensors(self, tensors) -> None:
            """Tests the allocate tensors functionality."""
            return None

        def execute(self, graph, context, inputs):
            """Tests the execute functionality."""
            return {}

    ep = DummyEP({"device_id": "1"})
    assert ep.device_id == 1
    assert ep.get_supported_nodes(Graph("x")) == []
    assert ep.allocate_tensors([]) is None
    ctx = ExecutionContext(SessionOptions())
    assert ep.execute(Graph("x"), ctx, {}) == {}


def test_serializer() -> None:
    """Tests the serializer functionality."""
    g = Graph("test")
    g.add_tensor(
        Tensor(
            "w",
            shape=(2, 2),
            dtype=DType.FLOAT32,
            is_initializer=True,
            data=b"1234123412341234",
        )
    )
    g.initializers.append("w")
    g.inputs.append("w")
    n = Node(
        "Relu",
        ["w"],
        ["out"],
        attributes={"test_float": Attribute("test_float", "FLOAT", 1.0)},
    )
    g.add_node(n)
    b = to_bytes(g)
    assert len(b) > 0
    with tempfile.NamedTemporaryFile(delete=False) as f:
        name = f.name
    save(g, name)
    assert os.path.getsize(name) > 0
    os.remove(name)


def test_dlpack() -> None:
    """Tests the dlpack functionality."""
    t = Constant("test", values=b"1234123412341234", shape=(2, 2), dtype=DType.FLOAT32)
    capsule = t.__dlpack__()
    assert capsule is not None
    assert t.__dlpack_device__() == (1, 0)
