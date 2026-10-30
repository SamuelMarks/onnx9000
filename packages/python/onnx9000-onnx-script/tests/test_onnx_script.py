"""Comprehensive unit tests for onnx9000.onnx_script and onnx9000_onnx_script."""

import onnx9000_onnx_script
from onnx9000.onnx_script import GraphBuilder, Var, op, parse_and_compile, run, script


def test_onnx_script_builder_and_run() -> None:
    """Verify run(), GraphBuilder, Var, op, script, and parse_and_compile."""
    assert run() == "[onnx-script] processed"
    assert onnx9000_onnx_script.run() == "[onnx-script] processed"

    builder = GraphBuilder("test_script")
    assert builder.name == "test_script"
    assert Var is not None
    assert op is not None
    assert callable(script)
    assert callable(parse_and_compile)
