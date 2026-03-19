from typing import Any

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Tensor
from onnx9000.core.serializer import serialize_model


class MockValueInfo:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


def test_serializer_metadata_and_vinfo():
    g = Graph("TestGraph")
    g.metadata_props["key1"] = "val1"

    t1 = Tensor("t1", (2, 2), DType.FLOAT32)
    g.tensors["t1"] = t1

    # string value_info
    g.value_info.append("t1")

    # object value_info
    v2 = MockValueInfo("v2", (1,), DType.INT32)
    g.value_info.append(v2)

    model_bytes = serialize_model(g)
    assert model_bytes is not None
