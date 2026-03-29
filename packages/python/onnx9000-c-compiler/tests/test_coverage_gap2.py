"""Tests for packages/python/onnx9000-c-compiler/tests/test_coverage_gap2.py."""

import struct

import pytest
from onnx9000.c_compiler.compiler import C89Compiler
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor


def test_compiler_misc():
    """Test compiler misc."""
    g = Graph("test")
    g.tensors["Y_Add"] = Tensor("Y_Add", shape=(1,))
    n = Node("Add", inputs=[], outputs=["Y_Add"])
    g.nodes.append(n)
    compiler = C89Compiler(g)
    compiler.generate()


def test_activations_extra():
    """Test activations extra."""
    g = Graph("test")
    g.tensors["X"] = Tensor("X", shape=(1, 10), dtype=DType.FLOAT32)
    g.tensors["Y_Softmax"] = Tensor("Y_Softmax", shape=(1, 10), dtype=DType.FLOAT32)
    g.nodes.append(Node("Softmax", inputs=["X"], outputs=["Y_Softmax"]))
    g.tensors["Y_LogSoftmax"] = Tensor("Y_LogSoftmax", shape=(1, 10), dtype=DType.FLOAT32)
    g.nodes.append(Node("LogSoftmax", inputs=["X"], outputs=["Y_LogSoftmax"]))
    g.tensors["Y_PRelu"] = Tensor("Y_PRelu", shape=(1, 10), dtype=DType.FLOAT32)
    g.nodes.append(Node("PRelu", inputs=["X"], outputs=["Y_PRelu"]))
    compiler = C89Compiler(g, use_math_h=True)
    (h, c) = compiler.generate()
    assert "Softmax" in c


def test_routing_extra():
    """Test routing extra."""
    g = Graph("test")
    g.tensors["X"] = Tensor("X", shape=(2, 3), dtype=DType.FLOAT32)
    g.tensors["Y_T"] = Tensor("Y_T", shape=(3, 2), dtype=DType.FLOAT32)
    g.nodes.append(
        Node(
            "Transpose",
            inputs=["X"],
            outputs=["Y_T"],
            attributes={"perm": Attribute("perm", value=[1, 0])},
        )
    )
    g.tensors["Y_Flat"] = Tensor("Y_Flat", shape=(6,), dtype=DType.FLOAT32)
    g.nodes.append(Node("Flatten", inputs=["X"], outputs=["Y_Flat"]))
    g.tensors["Y_PadDyn"] = Tensor("Y_PadDyn", shape=(10, 10), dtype=DType.FLOAT32)
    g.nodes.append(Node("Pad", inputs=["X"], outputs=["Y_PadDyn"]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "Flatten" in c
    assert "Transpose" in c


def test_spatial_extra():
    """Test spatial extra."""
    g = Graph("test")
    g.tensors["X1"] = Tensor("X1", shape=(1, 3, 10, 10), dtype=DType.FLOAT32)
    g.tensors["W1"] = Tensor("W1", shape=(6, 3, 1, 1), dtype=DType.FLOAT32)
    g.tensors["Y1"] = Tensor("Y1", shape=(1, 6, 10, 10), dtype=DType.FLOAT32)
    g.nodes.append(Node("Conv", inputs=["X1", "W1"], outputs=["Y1"]))
    g.tensors["X2"] = Tensor("X2", shape=(1, 3, 10), dtype=DType.FLOAT32)
    g.tensors["W2"] = Tensor("W2", shape=(3, 6, 3), dtype=DType.FLOAT32)
    g.tensors["Y2"] = Tensor("Y2", shape=(1, 6, 10), dtype=DType.FLOAT32)
    g.nodes.append(Node("ConvTranspose", inputs=["X2", "W2"], outputs=["Y2"]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "1x1 Conv optimized" in c


def test_pooling_extra():
    """Test pooling extra."""
    g = Graph("test")
    g.tensors["X1"] = Tensor("X1", shape=(1, 3, 10), dtype=DType.FLOAT32)
    g.tensors["Y_Avg1D"] = Tensor("Y_Avg1D", shape=(1, 3, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("AveragePool", inputs=["X1"], outputs=["Y_Avg1D"]))
    g.tensors["Y_Avg2D"] = Tensor("Y_Avg2D", shape=(1, 3, 5, 5), dtype=DType.FLOAT32)
    g.tensors["X2"] = Tensor("X2", shape=(1, 3, 10, 10), dtype=DType.FLOAT32)
    g.nodes.append(Node("AveragePool", inputs=["X2"], outputs=["Y_Avg2D"]))
    g.tensors["X3"] = Tensor("X3", shape=(1, 3, 10, 10, 10), dtype=DType.FLOAT32)
    g.tensors["Y_Max3D"] = Tensor("Y_Max3D", shape=(1, 3, 5, 5, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("MaxPool", inputs=["X3"], outputs=["Y_Max3D"]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()


def test_operations_extra():
    """Test operations extra."""
    g = Graph("test")
    g.tensors["X1"] = Tensor("X1", shape=(1, 50), dtype=DType.INT32)
    g.tensors["W1"] = Constant("W1", shape=(1,), dtype=DType.INT32, values=struct.pack("<i", 4))
    g.tensors["Y1"] = Tensor("Y1", shape=(1, 50), dtype=DType.INT32)
    g.nodes.append(Node("Div", inputs=["X1", "W1"], outputs=["Y1"]))
    g.tensors["M_A"] = Tensor("M_A", shape=(2, 2), dtype=DType.FLOAT32)
    g.tensors["M_B"] = Tensor("M_B", shape=(2, 2), dtype=DType.FLOAT32)
    g.tensors["M_C"] = Tensor("M_C", shape=(2, 2), dtype=DType.FLOAT32)
    g.nodes.append(Node("MatMul", inputs=["M_A", "M_B"], outputs=["M_C"]))
    g.tensors["V_A"] = Tensor("V_A", shape=(1, 40), dtype=DType.FLOAT32)
    g.tensors["V_B"] = Tensor("V_B", shape=(40, 50), dtype=DType.FLOAT32)
    g.tensors["V_C"] = Tensor("V_C", shape=(1, 50), dtype=DType.FLOAT32)
    g.nodes.append(Node("MatMul", inputs=["V_A", "V_B"], outputs=["V_C"]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "Optimized Div" in c
    assert "Small MatMul Unrolled" in c
    assert "GEMV Optimized" in c


def test_quantization_extra():
    """Test quantization extra."""
    g = Graph("test")
    g.tensors["X"] = Tensor("X", shape=(1, 50), dtype=DType.FLOAT32)
    g.tensors["S"] = Tensor("S", shape=(1,), dtype=DType.FLOAT32)
    g.tensors["ZP_NO"] = Tensor("ZP_NO", shape=(1,), dtype=DType.FLOAT32)
    g.tensors["Y_Q"] = Tensor("Y_Q", shape=(1, 50), dtype=DType.INT8)
    g.nodes.append(Node("QuantizeLinear", inputs=["X", "S"], outputs=["Y_Q"]))
    g.tensors["X_DQ"] = Tensor("X_DQ", shape=(1, 50), dtype=DType.INT8)
    g.tensors["Y_DQ"] = Tensor("Y_DQ", shape=(1, 50), dtype=DType.FLOAT32)
    g.nodes.append(Node("DequantizeLinear", inputs=["X_DQ", "S"], outputs=["Y_DQ"]))
    g.tensors["Q_A"] = Tensor("Q_A", shape=(2, 2), dtype=DType.INT8)
    g.tensors["Q_B"] = Tensor("Q_B", shape=(2, 2), dtype=DType.INT8)
    g.tensors["Q_C"] = Tensor("Q_C", shape=(2, 2), dtype=DType.INT8)
    g.nodes.append(
        Node(
            "QLinearMatMul",
            inputs=["Q_A", "S", "ZP_NO", "Q_B", "S", "ZP_NO", "S", "ZP_NO"],
            outputs=["Q_C"],
        )
    )
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
