import pytest
from onnx9000.c_compiler.compiler import C89Compiler
from onnx9000.core.ir import DType, Graph, Node, ValueInfo


def test_fuzzer_stub():
    """Mock test representing libFuzzer infrastructure."""
    g = Graph("fuzz")
    g.inputs.append(ValueInfo("X", (2,), DType.FLOAT32))
    g.outputs.append("Y")
    g.nodes.append(Node("Relu", ["X"], ["Y"]))

    comp = C89Compiler(g)
    h, c = comp.generate()

    # Assert bounds checking macros exist
    assert "CHECK_BOUNDS" in c
