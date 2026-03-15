import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import CompilationError
from onnx9000.core.parser.inference import infer_shapes_and_types, _INFERENCE_RULES


def _mock_graph(nodes):
    g = Graph("test")
    for n in nodes:
        g.add_node(n)
    return g


def test_inference_missing_lines():
    g = _mock_graph(
        [Node(op_type="ReduceMean", inputs=["A"], outputs=["Y"], attributes={})]
    )
    with pytest.raises(CompilationError, match="Input tensor not found."):
        infer_shapes_and_types(g)
    g = _mock_graph(
        [Node(op_type="MatMul", inputs=["A", "B"], outputs=["Y"], attributes={})]
    )
    with pytest.raises(CompilationError, match="Input tensor not found."):
        infer_shapes_and_types(g)
    g = _mock_graph(
        [Node(op_type="Gemm", inputs=["A", "B"], outputs=["Y"], attributes={})]
    )
    with pytest.raises(CompilationError, match="Input tensor not found."):
        infer_shapes_and_types(g)
    g = _mock_graph(
        [Node(op_type="SequenceMap", inputs=["A"], outputs=[], attributes={})]
    )
    with pytest.raises(CompilationError, match="SequenceMap expects 1 output."):
        infer_shapes_and_types(g)
    g = _mock_graph([Node(op_type="Constant", inputs=[], outputs=[], attributes={})])
    with pytest.raises(CompilationError, match="Constant expects exactly 1 output."):
        infer_shapes_and_types(g)
    g = _mock_graph(
        [Node(op_type="DeformConv", inputs=["A", "B"], outputs=["Y"], attributes={})]
    )
    with pytest.raises(CompilationError, match="DeformConv expects at least 3 inputs"):
        infer_shapes_and_types(g)
    g = _mock_graph(
        [Node(op_type="RandomNormal", inputs=[], outputs=[], attributes={})]
    )
    with pytest.raises(CompilationError, match="RandomNormal expects 1 output."):
        infer_shapes_and_types(g)
    g = _mock_graph(
        [Node(op_type="ArgMax", inputs=["A"], outputs=["Y"], attributes={})]
    )
    with pytest.raises(CompilationError, match="Input tensor A not found."):
        infer_shapes_and_types(g)
    g = _mock_graph(
        [
            Node(
                op_type="Attention",
                inputs=["A", "B", "C"],
                outputs=["", "", "", ""],
                attributes={},
            )
        ]
    )
    g.add_tensor(Tensor("A", (1, 2, 3), DType.FLOAT32))
    g.add_tensor(Tensor("B", (1, 2, 3), DType.FLOAT32))
    g.add_tensor(Tensor("C", (1, 2, 3), DType.FLOAT32))
    infer_shapes_and_types(g)


def test_deformconv_missing_tensors():
    g = _mock_graph(
        [
            Node(
                op_type="DeformConv",
                inputs=["A", "B", "C"],
                outputs=["Y"],
                attributes={},
            )
        ]
    )
    with pytest.raises(CompilationError, match="DeformConv inputs not found."):
        infer_shapes_and_types(g)
