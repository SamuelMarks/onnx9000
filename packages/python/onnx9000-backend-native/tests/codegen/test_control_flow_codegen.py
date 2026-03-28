"""Module providing functionality for test_control_flow_codegen."""

import pytest
from onnx9000.backends.codegen.generator import Generator
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor


def test_if_codegen():
    """Test if codegen."""
    g = Graph("TestIf")
    g.inputs = ["cond"]
    g.outputs = ["out"]
    g.tensors["cond"] = Tensor("cond", (1,), DType.BOOL)
    g.tensors["out"] = Tensor("out", (1,), DType.FLOAT32)

    then_g = Graph("Then")
    then_g.inputs = ["in1"]
    then_g.outputs = ["out1"]
    g.tensors["in1"] = Tensor("in1", (1,), DType.FLOAT32)
    g.tensors["out1"] = Tensor("out1", (1,), DType.FLOAT32)
    then_g.nodes.append(Node("Abs", inputs=["in1"], outputs=["out1"]))

    else_g = Graph("Else")
    else_g.inputs = ["in2"]
    else_g.outputs = ["out2"]
    g.tensors["in2"] = Tensor("in2", (1,), DType.FLOAT32)
    g.tensors["out2"] = Tensor("out2", (1,), DType.FLOAT32)
    else_g.nodes.append(Node("Neg", inputs=["in2"], outputs=["out2"]))

    n_if = Node("If", inputs=["cond"], outputs=["out"])
    n_if.attributes["then_branch"] = then_g
    n_if.attributes["else_branch"] = else_g
    g.nodes.append(n_if)

    gen = Generator(g)
    code = gen.generate()
    assert "Abs" in code
    assert "Neg" in code


def test_loop_codegen():
    """Test loop codegen."""
    g = Graph("TestLoop")
    g.inputs = ["M", "cond"]
    g.outputs = ["out"]
    g.tensors["M"] = Tensor("M", (1,), DType.INT64)
    g.tensors["cond"] = Tensor("cond", (1,), DType.BOOL)
    g.tensors["out"] = Tensor("out", (1,), DType.FLOAT32)

    body_g = Graph("Body")
    body_g.inputs = ["in1"]
    body_g.outputs = ["out1"]
    g.tensors["in1"] = Tensor("in1", (1,), DType.FLOAT32)
    g.tensors["out1"] = Tensor("out1", (1,), DType.FLOAT32)
    body_g.nodes.append(Node("Relu", inputs=["in1"], outputs=["out1"]))

    n_loop = Node("Loop", inputs=["M", "cond"], outputs=["out"])
    n_loop.attributes["body"] = body_g
    g.nodes.append(n_loop)

    gen = Generator(g)
    code = gen.generate()
    assert "Relu" in code
