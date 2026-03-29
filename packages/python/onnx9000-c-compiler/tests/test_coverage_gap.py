"""Tests for packages/python/onnx9000-c-compiler/tests/test_coverage_gap.py."""

import struct

import pytest
from onnx9000.c_compiler.compiler import C89Compiler
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, Node, Tensor


def test_ast_builder_edges():
    """Test ast builder edges."""
    from onnx9000.c_compiler.ast_builder import C89Builder

    b = C89Builder()
    assert b._sanitize("1name") == "v1name"
    b.push_indent()
    b.pop_indent()
    b.pop_indent()
    assert b.indent_level == 0


def test_unsupported_node():
    """Test unsupported node."""
    g = Graph("test")
    n = Node("UnsupportedOp", inputs=["X"], outputs=["Y"])
    g.nodes.append(n)
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "Unsupported node: UnsupportedOp" in c


def test_data_unpacker_edges():
    """Test data unpacker edges."""
    from onnx9000.c_compiler.data_unpacker import unpack_bytes_to_str

    res = unpack_bytes_to_str(b"\x01\x02\x03", DType.FLOAT32)
    assert "0x01, 0x02, 0x03" in res
    res64 = unpack_bytes_to_str(struct.pack("<2d", 1.5, 2.0), DType.FLOAT64)
    assert "1.5" in res64
    assert "2.0" in res64


def test_activations_coverage():
    """Test activations coverage."""
    g = Graph("test_activations")
    g.tensors["X"] = Tensor("X", shape=(1, 10), dtype=DType.FLOAT32)
    for op in ["LeakyRelu", "Sigmoid", "Tanh", "HardSigmoid", "HardSwish", "Gelu", "Clip", "PRelu"]:
        g.tensors[f"Y_{op}"] = Tensor(f"Y_{op}", shape=(1, 10), dtype=DType.FLOAT32)
        n = Node(op, inputs=["X"], outputs=[f"Y_{op}"])
        g.nodes.append(n)
    for norm in ["LayerNormalization", "InstanceNormalization"]:
        g.tensors[f"Y_{norm}"] = Tensor(f"Y_{norm}", shape=(1, 10), dtype=DType.FLOAT32)
        n = Node(norm, inputs=["X"], outputs=[f"Y_{norm}"])
        g.nodes.append(n)
    compiler = C89Compiler(g, use_math_h=False)
    (h, c) = compiler.generate()
    assert "LeakyRelu" in c
    assert "Sigmoid" in c
    assert "Tanh" in c
    assert "HardSigmoid" in c
    assert "HardSwish" in c
    assert "Gelu" in c
    assert "Clip" in c
    assert "LayerNormalization" in c
    assert "InstanceNormalization" in c


def test_boolean_coverage():
    """Test boolean coverage."""
    g = Graph("test_boolean")
    g.tensors["X1"] = Tensor("X1", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["X2"] = Tensor("X2", shape=(1, 5), dtype=DType.FLOAT32)
    for op in [
        "Equal",
        "Less",
        "LessOrEqual",
        "Greater",
        "GreaterOrEqual",
        "And",
        "Or",
        "Xor",
        "Not",
    ]:
        out = f"Y_{op}"
        g.tensors[out] = Tensor(out, shape=(1, 5), dtype=DType.BOOL)
        inputs = ["X1", "X2"] if op != "Not" else ["X1"]
        g.nodes.append(Node(op, inputs=inputs, outputs=[out]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "Xor" in c
    assert "LessOrEqual" in c
    assert "Greater" in c


def test_pooling_and_reduction_coverage():
    """Test pooling and reduction coverage."""
    g = Graph("test_pool")
    g.tensors["X2D"] = Tensor("X2D", shape=(1, 3, 10, 10), dtype=DType.FLOAT32)
    g.tensors["X1D"] = Tensor("X1D", shape=(1, 3, 10), dtype=DType.FLOAT32)
    g.tensors["Y_Avg2D"] = Tensor("Y_Avg2D", shape=(1, 3, 5, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("AveragePool", inputs=["X2D"], outputs=["Y_Avg2D"]))
    g.tensors["Y_Max1D"] = Tensor("Y_Max1D", shape=(1, 3, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("MaxPool", inputs=["X1D"], outputs=["Y_Max1D"]))
    g.tensors["Y_Avg1D"] = Tensor("Y_Avg1D", shape=(1, 3, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("AveragePool", inputs=["X1D"], outputs=["Y_Avg1D"]))
    for rop in ["ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd"]:
        out = f"Y_{rop}"
        g.tensors[out] = Tensor(out, shape=(1, 3), dtype=DType.FLOAT32)
        g.nodes.append(Node(rop, inputs=["X2D"], outputs=[out]))
        out_flat = f"Y_{rop}_flat"
        g.tensors[out_flat] = Tensor(out_flat, shape=(1,), dtype=DType.FLOAT32)
        from onnx9000.core.ir import Attribute

        g.nodes.append(
            Node(
                rop,
                inputs=["X2D"],
                outputs=[out_flat],
                attributes={"axes": Attribute("axes", value=[0, 1, 2, 3])},
            )
        )
    for rop in ["ArgMin"]:
        out = f"Y_{rop}"
        g.tensors[out] = Tensor(out, shape=(1, 3, 10), dtype=DType.INT64)
        g.nodes.append(Node(rop, inputs=["X2D"], outputs=[out]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "AveragePool" in c
    assert "ReduceMin" in c
    assert "ReduceProd" in c
    assert "ArgMin" in c


def test_routing_coverage():
    """Test routing coverage."""
    g = Graph("test_route")
    g.tensors["X3D"] = Tensor("X3D", shape=(2, 3, 4), dtype=DType.FLOAT32)
    g.tensors["Y_Trans3D"] = Tensor("Y_Trans3D", shape=(2, 4, 3), dtype=DType.FLOAT32)
    from onnx9000.core.ir import Attribute

    g.nodes.append(
        Node(
            "Transpose",
            inputs=["X3D"],
            outputs=["Y_Trans3D"],
            attributes={"perm": Attribute("perm", value=[0, 2, 1])},
        )
    )
    g.tensors["X4D"] = Tensor("X4D", shape=(2, 3, 4, 5), dtype=DType.FLOAT32)
    g.tensors["Y_Trans4D"] = Tensor("Y_Trans4D", shape=(5, 4, 3, 2), dtype=DType.FLOAT32)
    g.nodes.append(
        Node(
            "Transpose",
            inputs=["X4D"],
            outputs=["Y_Trans4D"],
            attributes={"perm": Attribute("perm", value=[3, 2, 1, 0])},
        )
    )
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "Unsupported Transpose dimensionality" in c
    assert "out_idx =" in c


def test_spatial_coverage():
    """Test spatial coverage."""
    g = Graph("test_spatial")
    g.tensors["X1"] = Tensor("X1", shape=(1, 3, 10), dtype=DType.FLOAT32)
    g.tensors["W1"] = Tensor("W1", shape=(6, 3, 3), dtype=DType.FLOAT32)
    g.tensors["Y1"] = Tensor("Y1", shape=(1, 6, 8), dtype=DType.FLOAT32)
    g.nodes.append(Node("Conv", inputs=["X1", "W1"], outputs=["Y1"]))
    g.tensors["X3"] = Tensor("X3", shape=(1, 3, 10, 10, 10), dtype=DType.FLOAT32)
    g.tensors["W3"] = Tensor("W3", shape=(6, 3, 3, 3, 3), dtype=DType.FLOAT32)
    g.tensors["Y3"] = Tensor("Y3", shape=(1, 6, 8, 8, 8), dtype=DType.FLOAT32)
    g.nodes.append(Node("Conv", inputs=["X3", "W3"], outputs=["Y3"]))
    g.tensors["XT1"] = Tensor("XT1", shape=(1, 3, 10), dtype=DType.FLOAT32)
    g.tensors["WT1"] = Tensor("WT1", shape=(3, 6, 3), dtype=DType.FLOAT32)
    g.tensors["YT1"] = Tensor("YT1", shape=(1, 6, 12), dtype=DType.FLOAT32)
    g.nodes.append(Node("ConvTranspose", inputs=["XT1", "WT1"], outputs=["YT1"]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "Conv1D" in c
    assert "Conv3D" in c
    assert "Unsupported ConvTranspose dim" in c
