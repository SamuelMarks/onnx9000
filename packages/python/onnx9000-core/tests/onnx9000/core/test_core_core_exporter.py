"""Unit tests for unified exporter in `onnx9000.core.exporter`."""

import os
import tempfile

from onnx9000.core.dtypes import DType
from onnx9000.core.exporter import (
    IRToONNXExporter,
    ONNXToKerasVisitor,
    export_graph,
    generate_keras,
    generate_mlir,
    register_exporter,
)
from onnx9000.core.ir import Graph, Node, ValueInfo


def test_exporter_functions() -> None:
    """Verify ONNXToKerasVisitor, IRToONNXExporter, register_exporter, export_graph, generate_keras, and generate_mlir."""
    g = Graph("exp_test")
    g.inputs.append(ValueInfo("x", (1, 3, 8, 8), DType.FLOAT32))
    g.outputs.append(ValueInfo("y", (1, 3, 8, 8), DType.FLOAT32))
    g.add_node(Node("Relu", ["x"], ["y"], name="r0"))

    keras_code = generate_keras(g)
    assert "def get_model_exp_test" in keras_code
    assert ONNXToKerasVisitor(g).generate() == keras_code

    mlir_code = generate_mlir(g)
    assert "func.func @main" in mlir_code

    called: list[str] = []

    @register_exporter("custom_fmt")
    def _custom_exp(graph: Graph, out: str) -> None:
        """Execute custom exp operation.

        Args:
            graph: Parameter specifying graph (Graph).
            out: Parameter specifying out (str).
        """
        called.append(out)

    export_graph(g, "out.custom", "custom_fmt")
    assert called == ["out.custom"]

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf:
        onnx_path = tf.name
    try:
        IRToONNXExporter(opset=17).export(g, onnx_path)
        assert os.path.exists(onnx_path)
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
