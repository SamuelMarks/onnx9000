"""Tests for packages/python/onnx9000-c-compiler/tests/test_coverage_final.py."""

import pytest
from onnx9000.c_compiler.compiler import C89Compiler
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor


def test_final_gaps():
    """Test final gaps."""
    g = Graph("final")
    g.tensors["A1"] = Tensor("A1", shape=("N", 4), dtype=DType.FLOAT32)
    g.tensors["A2"] = Tensor("A2", shape=(3, 1), dtype=DType.FLOAT32)
    g.tensors["AY"] = Tensor("AY", shape=(3, "N", 4), dtype=DType.FLOAT32)
    g.nodes.append(Node("Add", inputs=["A1", "A2"], outputs=["AY"]))
    g.nodes.append(Node("Sign", inputs=["MISSING"], outputs=["Y_MISSING"]))
    g.tensors["Y_MISSING"] = Tensor("Y_MISSING", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("MatMul", inputs=["MISS_A", "MISS_B"], outputs=["M_EY"]))
    g.tensors["M_EY"] = Tensor("M_EY", shape=(2, 2), dtype=DType.FLOAT32)
    g.tensors["R_X"] = Tensor("R_X", shape=(2, 3, 4), dtype=DType.FLOAT32)
    g.tensors["R_Y"] = Tensor("R_Y", shape=(2, 1, 4), dtype=DType.FLOAT32)
    g.nodes.append(
        Node(
            "ReduceSum",
            inputs=["R_X"],
            outputs=["R_Y"],
            attributes={"axes": Attribute("axes", value=[1])},
        )
    )
    g.tensors["Q_X"] = Tensor("Q_X", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["Q_S"] = Tensor("Q_S", shape=(1,), dtype=DType.FLOAT32)
    g.tensors["Q_Y"] = Tensor("Q_Y", shape=(1, 5), dtype=DType.INT8)
    g.nodes.append(Node("QuantizeLinear", inputs=["Q_X", "Q_S"], outputs=["Q_Y"]))
    g.tensors["DQ_X"] = Tensor("DQ_X", shape=(1, 5), dtype=DType.INT8)
    g.tensors["DQ_Y"] = Tensor("DQ_Y", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("DequantizeLinear", inputs=["DQ_X", "Q_S"], outputs=["DQ_Y"]))
    g.tensors["QC_X"] = Tensor("QC_X", shape=(1, 3, 10), dtype=DType.UINT8)
    g.tensors["QC_W"] = Tensor("QC_W", shape=(6, 3, 3), dtype=DType.UINT8)
    g.tensors["QC_Y"] = Tensor("QC_Y", shape=(1, 6, 8), dtype=DType.UINT8)
    g.nodes.append(
        Node(
            "QLinearConv",
            inputs=["QC_X", "Q_S", "Q_S", "QC_W", "Q_S", "Q_S", "Q_S", "Q_S"],
            outputs=["QC_Y"],
        )
    )
    g.tensors["DC_X"] = Tensor("DC_X", shape=(1, 64, 10, 10), dtype=DType.FLOAT32)
    g.tensors["DC_W"] = Tensor("DC_W", shape=(64, 1, 3, 3), dtype=DType.FLOAT32)
    g.tensors["DC_Y"] = Tensor("DC_Y", shape=(1, 64, 10, 10), dtype=DType.FLOAT32)
    g.nodes.append(
        Node(
            "Conv",
            inputs=["DC_X", "DC_W"],
            outputs=["DC_Y"],
            attributes={"group": Attribute("group", value=64)},
        )
    )
    c = C89Compiler(g)
    c.generate()


def test_layer_norm_no_scale():
    """Test layer norm no scale."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor

    g = Graph("test")
    g.tensors["X"] = Tensor("X", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["Y_LN"] = Tensor("Y_LN", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("LayerNormalization", inputs=["X"], outputs=["Y_LN"]))
    c = C89Compiler(g)
    c.generate()


def test_operations_gemv_no_alpha():
    """Test operations gemv no alpha."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor

    g = Graph("test")
    g.tensors["V_A"] = Tensor("V_A", shape=(1, 40), dtype=DType.FLOAT32)
    g.tensors["V_B"] = Tensor("V_B", shape=(40, 50), dtype=DType.FLOAT32)
    g.tensors["V_C"] = Tensor("V_C", shape=(1, 50), dtype=DType.FLOAT32)
    g.nodes.append(Node("Gemm", inputs=["V_A", "V_B", "V_C"], outputs=["V_C"]))
    c = C89Compiler(g)
    c.generate()


def test_quantization_no_bias():
    """Test quantization no bias."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor

    g = Graph("test")
    g.tensors["QC_X"] = Tensor("QC_X", shape=(1, 3, 10, 10), dtype=DType.UINT8)
    g.tensors["QC_W"] = Tensor("QC_W", shape=(6, 3, 3, 3), dtype=DType.UINT8)
    g.tensors["S"] = Tensor("S", shape=(1,), dtype=DType.FLOAT32)
    g.tensors["ZP"] = Tensor("ZP", shape=(1,), dtype=DType.UINT8)
    g.tensors["QC_Y"] = Tensor("QC_Y", shape=(1, 6, 8, 8), dtype=DType.UINT8)
    g.nodes.append(
        Node(
            "QLinearConv",
            inputs=["QC_X", "S", "ZP", "QC_W", "S", "ZP", "S", "ZP"],
            outputs=["QC_Y"],
        )
    )
    c = C89Compiler(g)
    c.generate()


def test_activations_no_scale_no_bias():
    """Test activations no scale no bias."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor

    g = Graph("test")
    g.tensors["X"] = Tensor("X", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("LayerNormalization", inputs=["X"], outputs=["Y"]))
    c = C89Compiler(g)
    c.generate()


def test_matmul_no_bias():
    """Test matmul no bias."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor

    g = Graph("test")
    g.tensors["A"] = Tensor("A", shape=(40, 50), dtype=DType.FLOAT32)
    g.tensors["B"] = Tensor("B", shape=(50, 40), dtype=DType.FLOAT32)
    g.tensors["C"] = Tensor("C", shape=(40, 40), dtype=DType.FLOAT32)
    g.nodes.append(
        Node(
            "Gemm", inputs=["A", "B"], outputs=["C"], attributes={"alpha": Attribute("alpha", 2.0)}
        )
    )
    c = C89Compiler(g)
    c.generate()


def test_qlinear_conv_no_bias():
    """Test qlinear conv no bias."""
    import struct

    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor

    g = Graph("test")
    g.tensors["QC_X"] = Tensor("QC_X", shape=(1, 3, 10, 10), dtype=DType.UINT8)
    g.tensors["QC_W"] = Tensor("QC_W", shape=(6, 3, 3, 3), dtype=DType.UINT8)
    g.tensors["S"] = Tensor("S", shape=(1,), dtype=DType.FLOAT32)
    g.tensors["ZP"] = Tensor("ZP", shape=(1,), dtype=DType.UINT8)
    g.tensors["QC_Y"] = Tensor("QC_Y", shape=(1, 6, 8, 8), dtype=DType.UINT8)
    g.nodes.append(
        Node(
            "QLinearConv",
            inputs=["QC_X", "S", "ZP", "QC_W", "S", "ZP", "S", "ZP"],
            outputs=["QC_Y"],
        )
    )
    c = C89Compiler(g)
    c.generate()


def test_all_remaining_gaps():
    """Test all remaining gaps."""
    from onnx9000.c_compiler.ast_builder import C89Builder
    from onnx9000.c_compiler.intrinsics import emit_esp_nn_qlinear_conv, emit_esp_nn_qlinear_matmul
    from onnx9000.core.ir import Node

    b = C89Builder()
    emit_esp_nn_qlinear_matmul(b, Node("test"), "i1", "i2", "o", 1, 1, 1, 1, 1, 1, 1.0)
    emit_esp_nn_qlinear_conv(
        b,
        Node("test"),
        "i1",
        "w",
        "b",
        "o",
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1.0,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
    )


def test_quant_no_target():
    """Test quant no target."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor

    g = Graph("test")
    g.tensors["Q_A"] = Tensor("Q_A", shape=(1, 2, 2), dtype=DType.INT8)
    g.tensors["S"] = Tensor("S", shape=(1,), dtype=DType.FLOAT32)
    g.tensors["ZP"] = Tensor("ZP", shape=(1,), dtype=DType.INT8)
    g.nodes.append(
        Node(
            "QLinearMatMul", inputs=["Q_A", "S", "ZP", "Q_A", "S", "ZP", "S", "ZP"], outputs=["Q_A"]
        )
    )
    g.tensors["QC_X"] = Tensor("QC_X", shape=(1, 3, 10, 10), dtype=DType.UINT8)
    g.tensors["QC_W"] = Tensor("QC_W", shape=(6, 3, 3, 3), dtype=DType.UINT8)
    g.nodes.append(
        Node(
            "QLinearConv",
            inputs=["QC_X", "S", "ZP", "QC_W", "S", "ZP", "S", "ZP"],
            outputs=["QC_X"],
        )
    )
    c = C89Compiler(g)
    c.generate()


def test_operations_gemv_no_bias():
    """Test operations gemv no bias."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor

    g = Graph("test")
    g.tensors["V_A"] = Tensor("V_A", shape=(1, 40), dtype=DType.FLOAT32)
    g.tensors["V_B"] = Tensor("V_B", shape=(40, 50), dtype=DType.FLOAT32)
    g.tensors["V_C"] = Tensor("V_C", shape=(1, 50), dtype=DType.FLOAT32)
    g.nodes.append(Node("Gemm", inputs=["V_A", "V_B"], outputs=["V_C"]))
    g.tensors["SCALAR_A"] = Tensor("SCALAR_A", shape=(), dtype=DType.FLOAT32)
    g.tensors["SCALAR_B"] = Tensor("SCALAR_B", shape=(), dtype=DType.FLOAT32)
    g.nodes.append(Node("Add", inputs=["SCALAR_A", "SCALAR_B"], outputs=["SCALAR_B"]))
    c = C89Compiler(g)
    c.generate()
