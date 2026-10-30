"""Unit tests for Core IR data structures in `onnx9000.core.ir`."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import (
    Attribute,
    Constant,
    DLDataType,
    DLDevice,
    DLManagedTensor,
    DLTensor,
    DynamicDim,
    Graph,
    Node,
    QuantizedTensor,
    SparseTensor,
    Tensor,
    ValueInfo,
    Variable,
)


def test_dlpack_and_ir_classes() -> None:
    """Verify DLPack ctypes structures, DynamicDim, Attribute, ValueInfo, Tensor, Node, and Graph."""
    dt = DLDataType(2, 32, 1)
    assert dt.bits == 32
    dev = DLDevice(1, 0)
    assert dev.device_type == 1
    dlt = DLTensor()
    assert dlt is not None
    dlm = DLManagedTensor()
    assert dlm is not None

    dim = DynamicDim("batch")
    assert str(dim) == "batch"
    assert dim == DynamicDim("batch")
    assert dim.to_dict() == {"value": "batch"}

    attr = Attribute("alpha", value=0.5)
    assert attr.attr_type == "FLOAT"
    assert attr.to_dict()["name"] == "alpha"

    vi = ValueInfo("x", (dim, 4), DType.FLOAT32)
    assert vi.name == "x"

    t = Tensor("t", (2, 2), DType.FLOAT32)
    assert t.shape == (2, 2)

    st = SparseTensor("st", dims=(2, 2), format="COO")
    assert st.format == "COO"

    var = Variable("v", (2, 2), DType.FLOAT32)
    assert var.name == "v"

    const = Constant("c", values=b"\x00" * 16, shape=(2, 2), dtype=DType.FLOAT32)
    assert const.is_initializer is True

    qt = QuantizedTensor("qt", type="QInt8", block_size=32, shape=(2, 2))
    assert qt.block_size == 32

    node = Node("Relu", inputs=[var], outputs=["y"], attributes={"a": attr}, name="relu_0")
    assert node.op_type == "Relu"

    g = Graph("ir_test")
    g.add_tensor(var)
    g.add_tensor(const)
    g.add_node(node)
    assert len(g.nodes) == 1
    assert g.to_dict()["name"] == "ir_test"
