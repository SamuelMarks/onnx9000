"""Tests the missing coverage module functionality."""

import os
import struct
import tempfile
from typing import NoReturn

import pytest
from onnx9000.core import onnx_pb2
from onnx9000.core.dtypes import DType
from onnx9000.core.execution import ExecutionContext, ExecutionProvider, SessionOptions
from onnx9000.core.ir import (
    Attribute,
    Constant,
    DynamicDim,
    Graph,
    Node,
    Tensor,
    ValueInfo,
)
from onnx9000.core.memory import MemoryMapError, mmap_tensor_data
from onnx9000.core.serializer import (
    SerializationError,
    _serialize_shape,
    save,
    serialize_model,
    to_bytes,
)
from onnx9000.core.shape_inference import infer_shapes_and_types
from onnx9000.core.symbolic import broadcast_shapes, evaluate_symbolic_expression
from onnx9000.core.utils import CyclicDependencyError, topological_sort


def test_execution_provider_defaults() -> None:
    """Tests the execution provider defaults functionality."""

    class BlankEP(ExecutionProvider):
        """Represents the BlankEP class and its associated logic."""

        def get_supported_nodes(self, graph):
            """Tests the get supported nodes functionality."""
            return super().get_supported_nodes(graph)

        def allocate_tensors(self, tensors):
            """Tests the allocate tensors functionality."""
            return super().allocate_tensors(tensors)

        def execute(self, graph, context, inputs):
            """Tests the execute functionality."""
            return super().execute(graph, context, inputs)

    ep = BlankEP({})
    assert ep.get_supported_nodes(Graph("t")) == []
    assert ep.allocate_tensors([]) is None
    assert ep.execute(Graph("t"), ExecutionContext(SessionOptions()), {}) == {}


def test_ir_str_repr() -> None:
    """Tests the ir str repr functionality."""
    d = DynamicDim("batch")
    assert str(d) == "batch"
    assert repr(d) == "DynamicDim(batch)"
    assert d != 5
    t = Constant("t", shape=(1,), dtype=DType.FLOAT32)
    assert repr(t).startswith("ir.Constant(name=t")
    assert t.__dlpack_device__() == (1, 0)
    with pytest.raises(ValueError):
        t.__dlpack__()
    t2 = Constant("t2", shape=(DynamicDim(-1),), dtype=DType.FLOAT32, values=b"123")
    with pytest.raises(ValueError):
        t2.__dlpack__()


def test_mmap_missing_file() -> None:
    """Tests the mmap missing file functionality."""
    with pytest.raises(MemoryMapError):
        mmap_tensor_data("non_existent_file.bin", 0, 100)


def test_mmap_out_of_bounds() -> None:
    """Tests the mmap out of bounds functionality."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"123")
        name = f.name
    with pytest.raises(MemoryMapError):
        mmap_tensor_data(name, 0, 10)
    os.remove(name)


def test_mmap_exception_handling(monkeypatch) -> None:
    """Tests the mmap exception handling functionality."""
    import mmap

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"123")
        name = f.name

    def mock_mmap(*args, **kwargs) -> NoReturn:
        """Tests the mock mmap functionality."""
        raise OSError("Mock error")

    monkeypatch.setattr(mmap, "mmap", mock_mmap)
    with pytest.raises(MemoryMapError):
        mmap_tensor_data(name, 0, 1)
    os.remove(name)


def test_symbolic_errors() -> None:
    """Tests the symbolic errors functionality."""
    with pytest.raises(Exception):
        broadcast_shapes((2,), (3,))


def test_utils_cyclic_dep() -> None:
    """Tests the utils cyclic dep functionality."""
    g = Graph("cyclic")
    g.add_node(Node("Identity", inputs=["A"], outputs=["B"], attributes={}))
    g.add_node(Node("Identity", inputs=["B"], outputs=["C"], attributes={}))
    g.add_node(Node("Identity", inputs=["C"], outputs=["A"], attributes={}))
    with pytest.raises(CyclicDependencyError):
        topological_sort(g)


def test_missing_symbolic() -> None:
    """Tests the missing symbolic functionality."""
    assert broadcast_shapes((1,), (5,)) == (5,)
    assert broadcast_shapes((5,), (1,)) == (5,)
    assert broadcast_shapes((DynamicDim(-1),), (5,)) == (5,)
    assert broadcast_shapes((5,), (DynamicDim(-1),)) == (5,)
    res = broadcast_shapes((DynamicDim("N"),), (5,))
    assert str(res[0]) == "max(N, 5)"
    assert evaluate_symbolic_expression("1 * B", {"A": 1}) == "1 * B"


def test_missing_execution() -> None:
    """Tests the missing execution functionality."""
    from onnx9000.core.execution import Environment

    e = Environment()
    e2 = Environment()
    assert e is e2
    assert Environment.get_device() == "CPU"


def test_missing_ir() -> None:
    """Tests the missing ir functionality."""
    a = Attribute("attr", "FLOAT", 1.0)
    assert repr(a) == "Attribute(name=attr, type=FLOAT, value=1.0)"
    vi = ValueInfo("V", (1,), DType.FLOAT32)
    assert repr(vi) == "ValueInfo(name=V, shape=(1,), dtype=DType.FLOAT32)"
    n = Node("Relu", ["X"], ["Y"], {})
    assert repr(n) == "ir.Node(Relu, ['X'] -> ['Y'])"
    g = Graph("g")
    assert g.get_node("non_existent") is None


def test_missing_utils() -> None:
    """Tests the missing utils functionality."""
    g = Graph("g")
    g.add_node(Node("Identity", ["X"], ["Y"], attributes={}, name="N1"))
    g.add_node(Node("Identity", ["Y"], ["Z"], attributes={}, name="N2"))
    from onnx9000.core.utils import topological_sort

    assert len(topological_sort(g)) == 2


def test_missing_shape_inference() -> None:
    """Tests the missing shape inference functionality."""
    g = Graph("g")
    g.add_node(Node("MissingOp", ["A", "B"], ["Y"], attributes={}))
    g.add_node(Node("Add", ["A"], ["Y"], attributes={}))
    g.add_node(Node("Add", ["X1", "X2"], ["Y"], attributes={}))
    g.add_node(Node("MatMul", ["A"], ["Y"], attributes={}))
    g.add_node(Node("MatMul", ["X1", "X2"], ["Y"], attributes={}))
    g.add_node(Node("Reshape", ["A"], ["Y"], attributes={}))
    g.add_node(Node("Relu", [], ["Y"], attributes={}))
    g.add_node(Node("Concat", ["A"], ["Y"], attributes={}))
    infer_shapes_and_types(g)


def test_missing_serializer() -> None:
    """Tests the missing serializer functionality."""
    g = Graph("g")
    g.doc_string = "doc"
    g.opset_imports["ai.onnx"] = 15
    t = Tensor("T1", (1,), DType.FLOAT32, data=struct.pack("<f", 1.0))
    g.add_tensor(t)
    t2 = Tensor("T2", (1,), DType.INT32, data=struct.pack("<i", 1))
    g.add_tensor(t2)
    t3 = Tensor("T3", (1,), DType.INT64, data=struct.pack("<q", 1))
    g.add_tensor(t3)
    g.initializers.extend(["T1", "T2", "T3"])
    g.inputs.extend(["T1"])
    g.outputs.extend(["T2"])
    v = ValueInfo("V_in", (1,), DType.FLOAT32)
    g.inputs.append(v)
    v2 = ValueInfo("V_out", (1,), DType.FLOAT32)
    g.outputs.append(v2)
    g.initializers.append("Missing_tensor")
    tp = onnx_pb2.TensorProto()
    gp = onnx_pb2.GraphProto()
    g.add_node(
        Node(
            "N",
            ["X"],
            ["Y"],
            attributes={
                "a_float": Attribute("a_float", "FLOAT", 1.0),
                "a_int": Attribute("a_int", "INT", 1),
                "a_string": Attribute("a_string", "STRING", "1"),
                "a_floats": Attribute("a_floats", "FLOATS", [1.0]),
                "a_ints": Attribute("a_ints", "INTS", [1]),
                "a_strings": Attribute("a_strings", "STRINGS", ["1"]),
                "a_tensor": Attribute("a_tensor", "TENSOR", tp),
                "a_graph": Attribute("a_graph", "GRAPH", gp),
                "a_unsupported": Attribute("a_unsupported", "UNKNOWN", None),
            },
        )
    )
    model_proto = serialize_model(g)
    assert model_proto is not None
    b = to_bytes(g)
    assert len(b) > 0
    with tempfile.NamedTemporaryFile(delete=False) as f:
        name = f.name
    save(g, name)
    os.remove(name)
    tshape = _serialize_shape((DynamicDim("N"), -1, DynamicDim(5), 1))
    assert tshape.dim[0].dim_param == "N"
    assert tshape.dim[1].dim_param == "?"
    assert tshape.dim[2].dim_value == 5
    assert tshape.dim[3].dim_value == 1


def test_serializer_error(monkeypatch) -> None:
    """Tests the serializer error functionality."""
    import onnx9000.core.serializer as s

    def mock_to_bytes(g) -> NoReturn:
        """Tests the mock to bytes functionality."""
        raise Exception("Mock error")

    monkeypatch.setattr(s, "to_bytes", mock_to_bytes)
    with pytest.raises(SerializationError):
        s.save(Graph("t"), "test.onnx")


def test_shape_inference_errors() -> None:
    """Tests the shape inference errors functionality."""
    from onnx9000.core.exceptions import ShapeInferenceError
    from onnx9000.core.shape_inference import infer_shapes_and_types

    g = Graph("cyclic")
    g.add_node(Node("Identity", inputs=["A"], outputs=["B"], attributes={}))
    g.add_node(Node("Identity", inputs=["B"], outputs=["C"], attributes={}))
    g.add_node(Node("Identity", inputs=["C"], outputs=["A"], attributes={}))
    with pytest.raises(ShapeInferenceError):
        infer_shapes_and_types(g)


def test_shape_inference_branches() -> None:
    """Tests the shape inference branches functionality."""
    from onnx9000.core.shape_inference import infer_shapes_and_types

    g = Graph("g")
    t1 = Tensor("I1", (1,), DType.FLOAT32)
    g.add_tensor(t1)
    g.inputs.append("I1")
    vi = ValueInfo("I2", (2,), DType.FLOAT32)
    g.inputs.append(vi)
    t2 = Tensor("T1", (1,), DType.FLOAT32, is_initializer=True)
    g.add_tensor(t2)
    g.add_node(Node("Add", ["I1", "MISSING2"], ["OUT1"], attributes={}))
    g.add_node(Node("MatMul", ["I1", "MISSING2"], ["OUT2"], attributes={}))
    infer_shapes_and_types(g)


def test_ir_find_node() -> None:
    """Tests the ir find node functionality."""
    g = Graph("g")
    n = Node("A", [], [], {}, name="N1")
    g.add_node(n)
    assert g.get_node("N1") is n
    g.print_visualizer()
