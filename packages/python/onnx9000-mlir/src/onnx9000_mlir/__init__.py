"""Backward-compatible re-export module for `onnx9000_mlir`."""

from onnx9000.mlir import Dialect, MLIRModule, MLIROp, MLIRType, MLIRValue, run

__all__ = ["Dialect", "MLIRModule", "MLIROp", "MLIRType", "MLIRValue", "run"]
