"""Backward-compatible re-export module for `onnx9000_mlir_lowering`."""

from onnx9000.mlir_lowering import MLIRLoweringPass, lower_to_linalg, lower_to_tosa, run

__all__ = ["MLIRLoweringPass", "lower_to_linalg", "lower_to_tosa", "run"]
