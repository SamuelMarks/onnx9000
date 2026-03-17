"""Module providing core logic and structural definitions."""

from onnx9000.converters.tf.importer import load_tf


def test_tf_importer_core_ops() -> None:
    """Tests the test_tf_importer_core_ops functionality."""
    graph_def = {
        "node": [
            {
                "op": "Placeholder",
                "name": "input_x",
                "attr": {
                    "dtype": 1,
                    "shape": {"dim": [{"size": 1}, {"size": 224}, {"size": 224}, {"size": 3}]},
                },
            },
            {"op": "Const", "name": "weights", "attr": {"dtype": 1}},
            {"op": "Conv2D", "name": "conv1", "input": ["input_x", "weights"]},
            {"op": "Relu", "name": "relu1", "input": ["conv1"]},
            {"op": "Const", "name": "mat_w", "attr": {"dtype": 1}},
            {"op": "MatMul", "name": "mat1", "input": ["relu1", "mat_w"]},
        ]
    }
    graph = load_tf(graph_def)
    op_types = [n.op_type for n in graph.nodes]
    assert "Conv" in op_types
    assert "Transpose" in op_types
    assert "Relu" in op_types
    assert "MatMul" in op_types
    assert len(graph.inputs) == 1
    assert len(graph.outputs) == 1
