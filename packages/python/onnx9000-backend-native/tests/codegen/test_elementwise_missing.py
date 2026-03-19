import pytest
from onnx9000.backends.codegen.generator import Generator
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor


def test_elementwise_missing():
    g = Graph("TestAllMissing")
    g.inputs = ["in1", "in2"]

    g.tensors["in1"] = Tensor("in1", (2, 2), DType.FLOAT32)
    g.tensors["in2"] = Tensor("in2", (2, 2), DType.FLOAT32)

    ops = [
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "Mod",
        "Erf",
        "IsInf",
        "IsNaN",
        "Floor",
        "Round",
        "Reciprocal",
        "Neg",
    ]

    for i, op in enumerate(ops):
        inps = (
            ["in1", "in2"]
            if op in ["Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Mod"]
            else ["in1"]
        )
        g.nodes.append(Node(op, inputs=inps, outputs=[f"out_{i}"]))
        g.tensors[f"out_{i}"] = Tensor(f"out_{i}", (2, 2), DType.FLOAT32)
        g.outputs.append(f"out_{i}")

    gen = Generator(g)
    code = gen.generate()

    assert "==" in code
    assert ">=" in code
    assert "<" in code
    assert "<=" in code
    assert "std::fmod" in code
    assert "std::erf" in code
    assert "std::isinf" in code
    assert "std::isnan" in code
    assert "std::floor" in code
    assert "std::round" in code
    assert "1.0f /" in code
    assert "-" in code
