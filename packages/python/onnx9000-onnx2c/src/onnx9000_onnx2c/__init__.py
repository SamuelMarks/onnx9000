"""Backward-compatible re-export module for `onnx9000_onnx2c`."""

from onnx9000.onnx2c import C89Compiler, compile_to_c, run

__all__ = ["C89Compiler", "compile_to_c", "run"]
