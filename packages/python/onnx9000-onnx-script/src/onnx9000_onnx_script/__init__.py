"""Backward-compatible re-export module for `onnx9000_onnx_script`."""

from onnx9000.onnx_script import GraphBuilder, Var, op, parse_and_compile, run, script

__all__ = ["GraphBuilder", "Var", "op", "parse_and_compile", "run", "script"]
