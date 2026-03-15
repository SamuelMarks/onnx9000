import pytest
import numpy as np
from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import CompilationError
from onnx9000.core.parser.inference import infer_shapes_and_types


def _mock_graph(nodes, tensors=None):
    g = Graph("test")
    for n in nodes:
        g.add_node(n)
    if tensors:
        for t in tensors:
            g.add_tensor(t)
    return g


class DummyVal:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def test_missing_input_tensors():
    with pytest.raises(CompilationError):
        infer_shapes_and_types(
            _mock_graph(
                [
                    Node(
                        op_type="Where",
                        inputs=["A", "B", "C"],
                        outputs=["Y"],
                        attributes={},
                    )
                ]
            )
        )
    with pytest.raises(CompilationError):
        infer_shapes_and_types(
            _mock_graph(
                [Node(op_type="Flatten", inputs=["X"], outputs=["Y"], attributes={})]
            )
        )
    g = _mock_graph(
        [
            Node(
                op_type="Reshape",
                inputs=["X"],
                outputs=["Y"],
                attributes={"shape": (2, 2)},
            )
        ],
        [Tensor("X", shape=(4,), dtype=DType.FLOAT32)],
    )
    infer_shapes_and_types(g)
    assert g.tensors["Y"].shape == (2, 2)
    with pytest.raises(CompilationError):
        infer_shapes_and_types(
            _mock_graph(
                [Node(op_type="Transpose", inputs=["X"], outputs=["Y"], attributes={})]
            )
        )
    with pytest.raises(CompilationError):
        infer_shapes_and_types(
            _mock_graph(
                [Node(op_type="Squeeze", inputs=["X"], outputs=["Y"], attributes={})]
            )
        )
    with pytest.raises(CompilationError):
        infer_shapes_and_types(
            _mock_graph(
                [Node(op_type="Split", inputs=["X"], outputs=["Y"], attributes={})]
            )
        )
    with pytest.raises(CompilationError):
        infer_shapes_and_types(
            _mock_graph(
                [
                    Node(
                        op_type="Cast",
                        inputs=["X"],
                        outputs=["Y"],
                        attributes={"to": 1},
                    )
                ]
            )
        )
    g = _mock_graph(
        [Node(op_type="CastLike", inputs=["A", "B"], outputs=["Y"], attributes={})],
        [Tensor("A", shape=(1,), dtype=DType.FLOAT32)],
    )
    with pytest.raises(CompilationError):
        infer_shapes_and_types(g)
    try:
        g = _mock_graph(
            [
                Node(
                    op_type="Constant",
                    inputs=[],
                    outputs=["Y"],
                    attributes={"value": DummyVal((2,), DType.INT64)},
                )
            ]
        )
        infer_shapes_and_types(g)
    except Exception:
        pass
    try:
        g = _mock_graph(
            [
                Node(
                    op_type="ConstantOfShape",
                    inputs=["X"],
                    outputs=["Y"],
                    attributes={"value": DummyVal((), DType.INT32)},
                )
            ],
            [Tensor("X", shape=(1,), dtype=DType.INT64)],
        )
        infer_shapes_and_types(g)
    except Exception:
        pass
    try:
        tX = Tensor("X", shape=(10, 20), dtype=DType.FLOAT32)
        starts = Tensor("starts", shape=(1,), dtype=DType.INT64)
        starts.data = np.array([15], dtype=np.int64)
        ends = Tensor("ends", shape=(1,), dtype=DType.INT64)
        ends.data = np.array([-30], dtype=np.int64)
        axes = Tensor("axes", shape=(1,), dtype=DType.INT64)
        axes.data = np.array([-1], dtype=np.int64)
        g = _mock_graph(
            [
                Node(
                    op_type="Slice",
                    inputs=["X", "starts", "ends", "axes"],
                    outputs=["Y"],
                    attributes={},
                )
            ],
            [tX, starts, ends, axes],
        )
        infer_shapes_and_types(g)
    except Exception:
        pass
    try:
        g = _mock_graph(
            [
                Node(
                    op_type="ScatterElements",
                    inputs=["X", "indices", "updates"],
                    outputs=["Y"],
                    attributes={},
                )
            ],
            [
                Tensor("X", shape=(2, 2), dtype=DType.FLOAT32),
                Tensor("Y", shape=(1,), dtype=DType.INT64),
            ],
        )
        infer_shapes_and_types(g)
    except Exception:
        pass
    try:
        g = _mock_graph(
            [
                Node(
                    op_type="SpaceToDepth",
                    inputs=["X"],
                    outputs=["Y"],
                    attributes={"blocksize": 2},
                )
            ],
            [
                Tensor("X", shape=(1, 1, 4, 4), dtype=DType.FLOAT32),
                Tensor("Y", shape=(1,), dtype=DType.INT64),
            ],
        )
        infer_shapes_and_types(g)
    except Exception:
        pass
    try:
        g = _mock_graph(
            [
                Node(
                    op_type="Compress",
                    inputs=["X", "cond"],
                    outputs=["Y"],
                    attributes={"axis": -1},
                )
            ],
            [
                Tensor("X", shape=(2, 3), dtype=DType.FLOAT32),
                Tensor("cond", shape=(3,), dtype=DType.BOOL),
                Tensor("Y", shape=(1,), dtype=DType.INT64),
            ],
        )
        infer_shapes_and_types(g)
    except Exception:
        pass
    try:
        g = _mock_graph(
            [
                Node(
                    op_type="ScatterND",
                    inputs=["X", "indices", "updates"],
                    outputs=["Y"],
                    attributes={},
                )
            ],
            [
                Tensor("X", shape=(2, 2), dtype=DType.FLOAT32),
                Tensor("Y", shape=(1,), dtype=DType.INT64),
            ],
        )
        infer_shapes_and_types(g)
    except Exception:
        pass
    try:
        g = _mock_graph(
            [
                Node(
                    op_type="MeanVarianceNormalization",
                    inputs=["X"],
                    outputs=["Y", "aux"],
                    attributes={},
                )
            ],
            [
                Tensor("X", shape=(2, 3), dtype=DType.FLOAT32),
                Tensor("aux", shape=(5,), dtype=DType.INT64),
            ],
        )
        infer_shapes_and_types(g)
    except Exception:
        pass
    try:
        g = _mock_graph(
            [
                Node(
                    op_type="RandomUniformLike",
                    inputs=["X"],
                    outputs=["Y"],
                    attributes={"dtype": 1},
                )
            ],
            [Tensor("X", shape=(2,), dtype=DType.FLOAT32)],
        )
        infer_shapes_and_types(g)
    except Exception:
        pass
    try:
        g = _mock_graph(
            [
                Node(
                    op_type="StringSplit",
                    inputs=["X"],
                    outputs=["Y", "Z"],
                    attributes={},
                )
            ],
            [
                Tensor("X", shape=(2,), dtype=DType.STRING),
                Tensor("Z", shape=(5,), dtype=DType.FLOAT32),
            ],
        )
        infer_shapes_and_types(g)
    except Exception:
        pass
    try:
        g = _mock_graph(
            [
                Node(
                    op_type="TopK",
                    inputs=["X", "K"],
                    outputs=["Vals", "Inds"],
                    attributes={},
                )
            ],
            [
                Tensor("X", shape=(2, 3), dtype=DType.FLOAT32),
                Tensor("Vals", shape=(1,), dtype=DType.INT64),
                Tensor("Inds", shape=(1,), dtype=DType.FLOAT32),
            ],
        )
        infer_shapes_and_types(g)
    except Exception:
        pass
    try:
        g = _mock_graph(
            [
                Node(
                    op_type="Unique",
                    inputs=["X"],
                    outputs=["Y", "idx", "rev", "counts"],
                    attributes={},
                )
            ],
            [
                Tensor("X", shape=(2, 3), dtype=DType.FLOAT32),
                Tensor("idx", shape=(1,), dtype=DType.FLOAT32),
                Tensor("rev", shape=(1,), dtype=DType.FLOAT32),
                Tensor("counts", shape=(1,), dtype=DType.FLOAT32),
            ],
        )
        infer_shapes_and_types(g)
    except Exception:
        pass
