"""Module providing core logic and structural definitions."""

import pytest
import numpy as np
from pathlib import Path
from onnx9000.core import onnx_pb2
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.core.exceptions import CompilationError
from onnx9000.core.parser.memory import plan_memory
from onnx9000.core.parser.passes import optimize
from onnx9000.core.parser.core import (
    _parse_dtype,
    _parse_shape,
    _parse_attribute,
    parse_model,
    load,
    from_bytes,
)


def test_plan_memory():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    t_in = Tensor("in1", (10,), DType.FLOAT32)
    t_init = Tensor("init1", (10,), DType.FLOAT32, is_initializer=True)
    t_out = Tensor("out1", (10,), DType.FLOAT32)
    t_out2 = Tensor("out2", (10,), DType.FLOAT32)
    g.inputs = ["in1"]
    g.initializers = ["init1"]
    g.outputs = ["out2"]
    g.add_tensor(t_in)
    g.add_tensor(t_init)
    g.add_tensor(t_out)
    g.add_tensor(t_out2)
    n1 = Node("Add", ["in1", "init1"], ["out1"], {}, "add1")
    n2 = Node("Relu", ["out1"], ["out2"], {}, "relu1")
    g.add_node(n1)
    g.add_node(n2)
    plan_memory(g)
    assert t_out.buffer_id is not None
    g_err = Graph("mock_err")
    g_err.inputs = []
    g_err.initializers = []
    t_bad = Tensor("bad", (10,), DType.FLOAT32)
    g_err.add_tensor(t_bad)
    n_err = Node("Relu", ["bad"], ["out_bad"], {}, "relu_err")
    g_err.add_node(n_err)
    with pytest.raises(CompilationError, match="used before creation"):
        plan_memory(g_err)


def test_optimize():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n1 = Node("Transpose", ["in1"], ["t1"], {"perm": [1, 0]}, "t1")
    n2 = Node("Transpose", ["t1"], ["out1"], {"perm": [1, 0]}, "t2")
    g.add_node(n1)
    g.add_node(n2)
    g.inputs = ["in1"]
    g.initializers = []
    g.outputs = ["out1"]
    g.add_tensor(Tensor("in1", (10, 10), DType.FLOAT32))
    g.add_tensor(Tensor("out1", (10, 10), DType.FLOAT32))
    optimize(g)


def test_parse_dtype():
    """Provides semantic functionality and verification."""
    assert _parse_dtype(1) == DType.FLOAT32
    with pytest.raises(CompilationError, match="Unsupported ONNX TensorProto"):
        _parse_dtype(999)


def test_parse_shape():
    """Provides semantic functionality and verification."""
    sp = onnx_pb2.TensorShapeProto()
    d1 = sp.dim.add()
    d1.dim_value = 10
    d2 = sp.dim.add()
    d2.dim_param = "batch"
    d3 = sp.dim.add()
    shape = _parse_shape(sp)
    assert shape[0] == 10
    assert shape[1] == DynamicDim("batch")
    assert shape[2] == DynamicDim(-1)


def test_parse_attribute():
    """Provides semantic functionality and verification."""
    a_f = onnx_pb2.AttributeProto(type=onnx_pb2.AttributeProto.FLOAT, f=1.5)
    assert _parse_attribute(a_f) == 1.5
    a_i = onnx_pb2.AttributeProto(type=onnx_pb2.AttributeProto.INT, i=42)
    assert _parse_attribute(a_i) == 42
    a_s = onnx_pb2.AttributeProto(type=onnx_pb2.AttributeProto.STRING, s=b"test")
    assert _parse_attribute(a_s) == "test"
    a_floats = onnx_pb2.AttributeProto(type=onnx_pb2.AttributeProto.FLOATS)
    a_floats.floats.extend([1.0, 2.0])
    assert _parse_attribute(a_floats) == [1.0, 2.0]
    a_ints = onnx_pb2.AttributeProto(type=onnx_pb2.AttributeProto.INTS)
    a_ints.ints.extend([1, 2])
    assert _parse_attribute(a_ints) == [1, 2]
    a_strings = onnx_pb2.AttributeProto(type=onnx_pb2.AttributeProto.STRINGS)
    a_strings.strings.extend([b"a", b"b"])
    assert _parse_attribute(a_strings) == ["a", "b"]
    a_t = onnx_pb2.AttributeProto(type=onnx_pb2.AttributeProto.TENSOR)
    t = onnx_pb2.TensorProto()
    a_t.t.CopyFrom(t)
    assert _parse_attribute(a_t) == t
    a_unk = onnx_pb2.AttributeProto(type=onnx_pb2.AttributeProto.GRAPH)
    assert _parse_attribute(a_unk) is None


def test_parse_model_and_io(tmp_path):
    """Provides semantic functionality and verification."""
    m = onnx_pb2.ModelProto()
    m.graph.name = "test"
    vi = m.graph.input.add()
    vi.name = "in"
    vi.type.tensor_type.elem_type = 1
    vi.type.tensor_type.shape.dim.add().dim_value = 10
    vo = m.graph.output.add()
    vo.name = "out"
    vo.type.tensor_type.elem_type = 1
    vo.type.tensor_type.shape.dim.add().dim_value = 10
    vi_val = m.graph.value_info.add()
    vi_val.name = "mid"
    vi_val.type.tensor_type.elem_type = 1
    vi_val.type.tensor_type.shape.dim.add().dim_value = 10
    init = m.graph.initializer.add()
    init.name = "w"
    init.data_type = 1
    init.dims.append(10)
    init.raw_data = np.ones(10, dtype=np.float32).tobytes()
    n = m.graph.node.add()
    n.op_type = "Add"
    n.input.extend(["in", "w"])
    n.output.extend(["out"])
    g = parse_model(m)
    assert g.name == "test"
    assert "in" in g.inputs
    assert "w" in g.initializers
    m_bytes = m.SerializeToString()
    g2 = from_bytes(m_bytes)
    assert g2.name == "test"
    f = tmp_path / "test.onnx"
    f.write_bytes(m_bytes)
    g3 = load(f)
    assert g3.name == "test"
