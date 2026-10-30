"""Comprehensive unit tests for onnx9000.mlir_lowering and onnx9000_mlir_lowering."""

import onnx9000_mlir_lowering
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.mlir_lowering import MLIRLoweringPass, lower_to_linalg, lower_to_tosa, run


def test_mlir_lowering_tosa_and_linalg() -> None:
    """Verify lower_to_tosa(), lower_to_linalg(), MLIRLoweringPass, and run()."""
    assert run() == "[mlir-lowering] processed"
    assert onnx9000_mlir_lowering.run() == "[mlir-lowering] processed"

    g = Graph("lower_test")
    x = Tensor("X", (1, 8), DType.FLOAT32)
    y = Tensor("Y", ("batch", 8), DType.FLOAT32)
    g.add_node(Node("Relu", [x], [y]))
    g.add_node(Node("Add", ["Y", "X"], ["Z"]))

    tosa_mod = lower_to_tosa(g)
    emitted_tosa = tosa_mod.emit()
    assert '"tosa.clamp"(%X) {min_fp = 0.0}' in emitted_tosa
    assert '"tosa.add"(%Y, %X)' in emitted_tosa

    linalg_mod = lower_to_linalg(g)
    emitted_linalg = linalg_mod.emit()
    assert '"linalg.clamp"(%X)' in emitted_linalg
    assert isinstance(MLIRLoweringPass(), MLIRLoweringPass)
