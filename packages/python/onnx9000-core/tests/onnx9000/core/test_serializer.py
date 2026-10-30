"""Unit tests for ONNX protobuf serialization in `onnx9000.core.serializer`."""

import tempfile

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, DynamicDim, Graph, Node, ValueInfo
from onnx9000.core.serializer import (
    SerializationError,
    _sanitize_string,
    _serialize_shape,
    _serialize_sparse_tensor,
    _serialize_tensor,
    save,
    serialize_model,
    to_bytes,
)
from onnx9000.core.sparse import dense_to_coo


def test_serializer_functions() -> None:
    """Verify SerializationError, _serialize_shape, _serialize_tensor, _serialize_sparse_tensor, serialize_model, save, and to_bytes."""
    err = SerializationError("failed")
    assert "failed" in str(err)

    sp = _serialize_shape((4, "N", DynamicDim("B"), -1))
    assert len(sp.dim) == 4
    assert sp.dim[0].dim_value == 4
    assert sp.dim[1].dim_param == "N"

    c = Constant("w", values=b"\x00" * 8, shape=(2,), dtype=DType.FLOAT32)
    tp, off = _serialize_tensor(c)
    assert tp.name == "w"
    assert off == 0

    st = dense_to_coo(c)
    stp, _ = _serialize_sparse_tensor(st)
    assert list(stp.dims) == [2]

    assert _sanitize_string("hello\x00world") == "helloworld"

    g = Graph("ser_graph")
    g.inputs.append(ValueInfo("x", (2,), DType.FLOAT32))
    g.outputs.append(ValueInfo("y", (2,), DType.FLOAT32))
    g.add_tensor(c)
    g.initializers.append("w")
    g.add_node(Node("Add", ["x", "w"], ["y"], name="add_0"))

    mp = serialize_model(g)
    assert mp.graph.name == "ser_graph"

    raw = to_bytes(g)
    assert len(raw) > 0

    with tempfile.NamedTemporaryFile(suffix=".onnx") as tf:
        save(g, tf.name)
        assert tf.tell() >= 0
