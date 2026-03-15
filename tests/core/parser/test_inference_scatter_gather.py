import pytest
from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.core.dtypes import DType
from onnx9000.core.parser.inference import infer_scatter_nd, infer_gather_nd


def test_scatter_nd_existing_output():
    g = Graph("test")
    g.add_tensor(Tensor("data", (2, 3), DType.FLOAT32))
    n1 = Node(
        op_type="ScatterND",
        inputs=["data", "indices", "updates"],
        outputs=["out_new"],
        attributes={},
    )
    infer_scatter_nd(n1, g)
    assert g.tensors["out_new"].shape == (2, 3)
    g.add_tensor(Tensor("out_existing", (1, 1), DType.FLOAT32))
    n2 = Node(
        op_type="ScatterND",
        inputs=["data", "indices", "updates"],
        outputs=["out_existing"],
        attributes={},
    )
    infer_scatter_nd(n2, g)
    assert g.tensors["out_existing"].shape == (2, 3)


def test_gather_nd():
    g = Graph("test_gather")
    g.add_tensor(Tensor("data", (2, 3), DType.FLOAT32))
    g.add_tensor(Tensor("indices", (2, 2), DType.INT64))
    n1 = Node(
        op_type="GatherND",
        inputs=["data", "indices"],
        outputs=["out_new"],
        attributes={"batch_dims": 1},
    )
    infer_gather_nd(n1, g)
    assert g.tensors["out_new"].shape[0].value == -1
    g.add_tensor(Tensor("out_existing", (1, 1), DType.FLOAT32))
    n2 = Node(
        op_type="GatherND",
        inputs=["data", "indices"],
        outputs=["out_existing"],
        attributes={"batch_dims": 1},
    )
    infer_gather_nd(n2, g)
    assert g.tensors["out_existing"].shape[0].value == -1
