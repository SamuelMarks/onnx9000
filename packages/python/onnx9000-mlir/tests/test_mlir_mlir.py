"""Comprehensive unit tests for onnx9000.mlir and onnx9000_mlir."""

import onnx9000_mlir
from onnx9000.mlir import Dialect, MLIRModule, MLIROp, MLIRType, MLIRValue, run


def test_mlir_ast_and_emission() -> None:
    """Verify MLIRType, MLIRValue, MLIROp, MLIRModule, and run()."""
    assert run() == "[mlir] processed"
    assert onnx9000_mlir.run() == "[mlir] processed"

    scalar_ty = MLIRType("f32")
    assert scalar_ty.format() == "f32"
    tensor_ty = MLIRType("f32", (1, -1, 4))
    assert tensor_ty.format() == "tensor<1x?x4xf32>"

    v_in = MLIRValue("arg0", tensor_ty)
    v_out = MLIRValue("res0", tensor_ty)
    op = MLIROp(
        dialect=Dialect.TOSA,
        name="clamp",
        operands=[v_in],
        results=[v_out],
        attributes={"min_fp": 0.0},
    )
    void_op = MLIROp(dialect=Dialect.FUNC, name="return", operands=[v_out])

    mod = MLIRModule("test_graph")
    mod.add_op(op)
    mod.add_op(void_op)
    text = mod.emit()
    assert "module @test_graph {" in text
    assert (
        '%res0 = "tosa.clamp"(%arg0) {min_fp = 0.0} : (tensor<1x?x4xf32>) -> (tensor<1x?x4xf32>)'
        in text
    )
    assert '"func.return"(%res0)' in text
