"""Module docstring."""

from onnx9000.ir import Graph, Node, Tensor
from onnx9000.dtypes import DType
import numpy as np


def test_slice_inference_coverage():
    """test_slice_inference_coverage docstring."""
    from onnx9000.parser.inference import infer_slice

    graph = Graph(name="test_slice")
    graph.add_tensor(Tensor(name="data", shape=(10, 20), dtype=DType.FLOAT32))

    starts_data = np.array([-5, -100, 100], dtype=np.int64)
    ends_data = np.array([-2, -100, 100], dtype=np.int64)
    axes_data = np.array([0, 0, 0], dtype=np.int64)
    steps_data = np.array([1, 1, 1], dtype=np.int64)

    graph.add_tensor(
        Tensor(
            name="starts",
            shape=(3,),
            dtype=DType.INT64,
            is_initializer=True,
            data=starts_data,
        )
    )
    graph.add_tensor(
        Tensor(
            name="ends",
            shape=(3,),
            dtype=DType.INT64,
            is_initializer=True,
            data=ends_data,
        )
    )
    graph.add_tensor(
        Tensor(
            name="axes",
            shape=(3,),
            dtype=DType.INT64,
            is_initializer=True,
            data=axes_data,
        )
    )
    graph.add_tensor(
        Tensor(
            name="steps",
            shape=(3,),
            dtype=DType.INT64,
            is_initializer=True,
            data=steps_data,
        )
    )

    node = Node(
        op_type="Slice",
        inputs=["data", "starts", "ends", "axes", "steps"],
        outputs=["out"],
        attributes={},
    )

    infer_slice(node, graph)


def test_compress_inference_coverage():
    """test_compress_inference_coverage docstring."""
    from onnx9000.parser.inference import infer_compress

    graph = Graph(name="test_compress")
    graph.add_tensor(Tensor(name="data", shape=(10, 20), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="cond", shape=(10,), dtype=DType.BOOL))

    node = Node(
        op_type="Compress",
        inputs=["data", "cond"],
        outputs=["out"],
        attributes={"axis": -1},
    )
    infer_compress(node, graph)
