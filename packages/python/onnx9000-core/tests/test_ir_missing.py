import ctypes

import pytest
from onnx9000.core.ir import *


def test_ir_missing() -> None:
    assert Attribute.infer_type(1) == "INT"
    assert Attribute.infer_type(1.0) == "FLOAT"
    assert Attribute.infer_type("test") == "STRING"
    assert Attribute.infer_type(Tensor("t")) == "TENSOR"
    assert Attribute.infer_type(Graph("g")) == "GRAPH"
    assert Attribute.infer_type([]) == "INTS"
    assert Attribute.infer_type([1]) == "INTS"
    assert Attribute.infer_type([1.0]) == "FLOATS"
    assert Attribute.infer_type(["test"]) == "STRINGS"
    assert Attribute.infer_type([Tensor("t")]) == "TENSORS"
    assert Attribute.infer_type([Graph("g")]) == "GRAPHS"
    assert Attribute.infer_type(object()) == "UNKNOWN"
    a1 = Attribute("a", value=1)
    a2 = Attribute("a", value=1)
    assert a1 == a2
    assert a1 != "a"
    t = Tensor("t")
    t.copy()
    n = Node("Op", inputs=[t], outputs=[])
    t.outputs.append(n)
    t.clear_inputs()
    t.clear_outputs()
    v = Variable("v", shape=(DynamicDim("N"), 2))
    assert v.is_dynamic()
    assert not v.is_empty()
    v2 = Variable("v2", shape=())
    assert v2.is_empty()
    c = Constant("c", values=b"abc", shape=(3,))
    assert c.values == b"abc"
    c.values = b"def"
    assert c.values == b"def"
    assert c.__dlpack_device__() == (1, 0)
    capsule = c.__dlpack__()
    assert capsule is not None
    ctypes.pythonapi.PyCapsule_IsValid.restype = ctypes.c_int
    ctypes.pythonapi.PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]
    assert ctypes.pythonapi.PyCapsule_IsValid(capsule, b"dltensor") == 1
    c_bad = Constant("c_bad", shape=(DynamicDim("N"),), values=b"a")
    with pytest.raises(ValueError):
        c_bad.__dlpack__()
    c_bad2 = Constant("c_bad2", shape=(-1,), values=b"a")
    with pytest.raises(ValueError):
        c_bad2.__dlpack__()
    c_empty = Constant("c_empty")
    with pytest.raises(ValueError):
        c_empty.__dlpack__()
    n2 = Node("Op", inputs=["in1"], outputs=["out1"])
    assert n2.i(0) == "in1"
    assert n2.o(0) == "out1"
    assert n2.copy().op == "Op"
    n2.op = "Op2"
    assert n2.op == "Op2"
    assert n2.attrs == {}
    assert n2 != "string"
    g = Graph("g")
    g2 = Graph("g")
    assert g == g2
    assert g != "string"
    g.add_tensor(v)
    g.add_node(n)
    assert g != g2
    g_copy = g.copy()
    assert Graph.tensors(g_copy) is not None
