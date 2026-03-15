import pytest
import numpy as np
from onnx9000.export.proto_utils import (
    to_tensor_proto,
    to_value_info_proto,
    to_attribute_proto,
    to_node_proto,
)
from onnx9000.frontends.frontend.tensor import Tensor, Parameter, Node
from onnx9000.core.dtypes import DType


def test_to_tensor_proto():
    p = Parameter(name="p1", data=np.array([1, 2, 3], dtype=np.int32))
    proto = to_tensor_proto(p)
    assert proto.name == "p1"
    assert proto.raw_data is not None


def test_to_value_info_proto():
    t = Tensor(name="t1", shape=(1, "batch_size"), dtype=DType.FLOAT32)
    proto = to_value_info_proto(t)
    assert proto.name == "t1"


def test_to_attribute_proto():
    attr = to_attribute_proto("f", 1.0)
    assert attr.f == 1.0
    attr = to_attribute_proto("i", 1)
    assert attr.i == 1
    attr = to_attribute_proto("s", "test")
    assert attr.s == b"test"
    attr = to_attribute_proto("ints", [1, 2])
    assert len(attr.ints) == 2
    attr = to_attribute_proto("floats", [1.0, 2.0])
    assert len(attr.floats) == 2
    attr = to_attribute_proto("strings", ["a", "b"])
    assert len(attr.strings) == 2
    with pytest.raises(ValueError):
        to_attribute_proto("bad_list", [object()])
    with pytest.raises(ValueError):
        to_attribute_proto("bad", object())


def test_to_node_proto():
    t_in = Tensor(name="in", shape=(1,), dtype=DType.FLOAT32)
    t_out = Tensor(name="out", shape=(1,), dtype=DType.FLOAT32)
    n = Node(op_type="Relu", inputs=[t_in], outputs=[t_out], attributes={"attr1": 1})
    proto = to_node_proto(n)
    assert proto.op_type == "Relu"
    assert proto.input[0] == "in"
    assert proto.output[0] == "out"
    assert proto.attribute[0].name == "attr1"
