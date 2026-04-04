"""Tests for packages/python/onnx9000-c-compiler/tests/test_coverage_gap3.py."""

import struct

from onnx9000.c_compiler.compiler import C89Compiler
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor


def test_ast_builder_edges_2():
    """Test ast builder edges 2."""
    from onnx9000.c_compiler.ast_builder import C89Builder

    b = C89Builder()
    b.pop_indent()
    assert b.indent_level == 0


def test_control_flow_coverage():
    """Test control flow coverage."""
    g = Graph("test")
    g.tensors["Cond"] = Tensor("Cond", shape=(1,), dtype=DType.BOOL)
    g.tensors["MaxCount"] = Tensor("MaxCount", shape=(1,), dtype=DType.INT64)

    class DummyGraph:
        """DummyGraph implementation."""

        name = "subgraph"

    n1 = Node(
        "If",
        inputs=["Cond"],
        outputs=[],
        attributes={
            "then_branch": Attribute("then_branch", value=DummyGraph()),
            "else_branch": Attribute("else_branch", value=DummyGraph()),
        },
    )
    n2 = Node(
        "Loop",
        inputs=["MaxCount", "Cond"],
        outputs=[],
        attributes={"body": Attribute("body", value=DummyGraph())},
    )
    g.nodes.extend([n1, n2])
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "Invoke then branch graph" in c
    assert "Invoke loop body graph" in c


def test_boolean_coverage():
    """Test boolean coverage."""
    g = Graph("test")
    g.tensors["X1"] = Tensor("X1", shape=(1, 5), dtype=DType.BOOL)
    g.tensors["X2"] = Tensor("X2", shape=(1, 5), dtype=DType.BOOL)
    g.tensors["XF1"] = Tensor("XF1", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["XF2"] = Tensor("XF2", shape=(1, 5), dtype=DType.FLOAT32)
    ops = [
        "Equal",
        "Less",
        "LessOrEqual",
        "Greater",
        "GreaterOrEqual",
        "And",
        "Or",
        "Xor",
        "Not",
        "Where",
    ]
    for op in ops:
        out = f"Y_{op}"
        g.tensors[out] = Tensor(out, shape=(1, 5), dtype=DType.BOOL)
        if op == "Not":
            g.nodes.append(Node(op, inputs=["X1"], outputs=[out]))
        elif op == "Where":
            g.nodes.append(Node(op, inputs=["X1", "XF1", "XF2"], outputs=[out]))
        elif op in ["And", "Or", "Xor"]:
            g.nodes.append(Node(op, inputs=["X1", "X2"], outputs=[out]))
        else:
            g.nodes.append(Node(op, inputs=["XF1", "XF2"], outputs=[out]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "Equal" in c
    assert "Where" in c


def test_pooling_and_reduction_more():
    """Test pooling and reduction more."""
    g = Graph("test")
    g.tensors["X_1D"] = Tensor("X1D", shape=(1, 3, 10), dtype=DType.FLOAT32)
    g.tensors["Y_Avg1D"] = Tensor("Y_Avg1D", shape=(1, 3, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("AveragePool", inputs=["X_1D"], outputs=["Y_Avg1D"]))
    g.tensors["X_3D"] = Tensor("X3D", shape=(1, 3, 10, 10, 10), dtype=DType.FLOAT32)
    g.tensors["Y_Max3D"] = Tensor("Y_Max3D", shape=(1, 3, 5, 5, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("MaxPool", inputs=["X_3D"], outputs=["Y_Max3D"]))
    g.tensors["X2D"] = Tensor("X2D", shape=(1, 3, 10, 10), dtype=DType.FLOAT32)
    for rop in [
        "ReduceMean",
        "ReduceSum",
        "ReduceMax",
        "ReduceMin",
        "ReduceProd",
        "ArgMax",
        "ArgMin",
    ]:
        out = f"Y_{rop}"
        g.tensors[out] = Tensor(out, shape=(1, 3, 10), dtype=DType.FLOAT32)
        g.nodes.append(
            Node(
                rop,
                inputs=["X2D"],
                outputs=[out],
                attributes={
                    "axes": Attribute("axes", value=[3]),
                    "axis": Attribute("axis", value=3),
                },
            )
        )
    g.tensors["Y_GlobalMax"] = Tensor("Y_GlobalMax", shape=(1, 3, 1, 1), dtype=DType.FLOAT32)
    g.nodes.append(Node("GlobalMaxPool", inputs=["X2D"], outputs=["Y_GlobalMax"]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "ReduceMean" in c
    assert "ArgMax" in c


def test_routing_more():
    """Test routing more."""
    g = Graph("test")
    g.tensors["X2D"] = Tensor("X2D", shape=(3, 4), dtype=DType.FLOAT32)
    g.tensors["X2D_b"] = Tensor("X2D_b", shape=(3, 4), dtype=DType.FLOAT32)
    g.tensors["Y_Concat"] = Tensor("Y_Concat", shape=(6, 4), dtype=DType.FLOAT32)
    g.nodes.append(Node("Concat", inputs=["X2D", "X2D_b"], outputs=["Y_Concat"]))
    g.tensors["Y_Squeeze"] = Tensor("Y_Squeeze", shape=(3, 4), dtype=DType.FLOAT32)
    g.nodes.append(Node("Squeeze", inputs=["X2D"], outputs=["Y_Squeeze"]))
    g.tensors["Y_Unsqueeze"] = Tensor("Y_Unsqueeze", shape=(1, 3, 4), dtype=DType.FLOAT32)
    g.nodes.append(Node("Unsqueeze", inputs=["X2D"], outputs=["Y_Unsqueeze"]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "Concat" in c
    assert "Squeeze" in c


def test_spatial_more():
    """Test spatial more."""
    g = Graph("test")
    g.tensors["X2D"] = Tensor("X2D", shape=(1, 64, 10, 10), dtype=DType.FLOAT32)
    g.tensors["W_Depth"] = Tensor("W_Depth", shape=(64, 1, 3, 3), dtype=DType.FLOAT32)
    g.tensors["Y_Depth"] = Tensor("Y_Depth", shape=(1, 64, 10, 10), dtype=DType.FLOAT32)
    g.nodes.append(
        Node(
            "Conv",
            inputs=["X2D", "W_Depth"],
            outputs=["Y_Depth"],
            attributes={"group": Attribute("group", value=64)},
        )
    )
    g.tensors["X_1D"] = Tensor("X1D", shape=(1, 3, 10), dtype=DType.FLOAT32)
    g.tensors["W_1D"] = Tensor("W_1D", shape=(6, 3, 3), dtype=DType.FLOAT32)
    g.tensors["Y_1D"] = Tensor("Y_1D", shape=(1, 6, 8), dtype=DType.FLOAT32)
    g.nodes.append(Node("Conv", inputs=["X_1D", "W_1D"], outputs=["Y_1D"]))
    g.tensors["X_3D"] = Tensor("X3D", shape=(1, 3, 10, 10, 10), dtype=DType.FLOAT32)
    g.tensors["W_3D"] = Tensor("W_3D", shape=(6, 3, 3, 3, 3), dtype=DType.FLOAT32)
    g.tensors["Y_3D"] = Tensor("Y_3D", shape=(1, 6, 8, 8, 8), dtype=DType.FLOAT32)
    g.nodes.append(Node("Conv", inputs=["X_3D", "W_3D"], outputs=["Y_3D"]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "Conv1D" in c
    assert "Conv3D" in c


def test_qlinear_conv():
    """Test qlinear conv."""
    g = Graph("test")
    g.tensors["X_q"] = Tensor("X_q", shape=(1, 3, 10, 10), dtype=DType.UINT8)
    g.tensors["W_q"] = Tensor("W_q", shape=(6, 3, 3, 3), dtype=DType.UINT8)
    g.tensors["S"] = Tensor("S", shape=(1,), dtype=DType.FLOAT32)
    g.tensors["ZP"] = Tensor("ZP", shape=(1,), dtype=DType.UINT8)
    g.tensors["Y_q"] = Tensor("Y_q", shape=(1, 6, 8, 8), dtype=DType.UINT8)
    g.nodes.append(
        Node("QLinearConv", inputs=["X_q", "S", "ZP", "W_q", "S", "ZP", "S", "ZP"], outputs=["Y_q"])
    )
    g.tensors["X_q1d"] = Tensor("X_q1d", shape=(1, 3, 10), dtype=DType.UINT8)
    g.tensors["W_q1d"] = Tensor("W_q1d", shape=(6, 3, 3), dtype=DType.UINT8)
    g.tensors["Y_q1d"] = Tensor("Y_q1d", shape=(1, 6, 8), dtype=DType.UINT8)
    g.nodes.append(
        Node(
            "QLinearConv",
            inputs=["X_q1d", "S", "ZP", "W_q1d", "S", "ZP", "S", "ZP"],
            outputs=["Y_q1d"],
        )
    )
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "QLinearConv" in c


def test_missing_ops():
    """Test missing ops."""
    g = Graph("test_ops")
    g.tensors["X3D_A"] = Tensor("X3D_A", shape=(2, 3, 4), dtype=DType.FLOAT32)
    g.tensors["X3D_B"] = Tensor("X3D_B", shape=(2, 4, 5), dtype=DType.FLOAT32)
    g.tensors["Y_BM"] = Tensor("Y_BM", shape=(2, 3, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("MatMul", inputs=["X3D_A", "X3D_B"], outputs=["Y_BM"]))
    g.tensors["X_IntA"] = Tensor("X_IntA", shape=(2, 2), dtype=DType.INT32)
    g.tensors["X_IntB"] = Tensor("X_IntB", shape=(2, 2), dtype=DType.INT32)
    g.tensors["Y_Int"] = Tensor("Y_Int", shape=(2, 2), dtype=DType.INT32)
    g.nodes.append(Node("MatMulInteger", inputs=["X_IntA", "X_IntB"], outputs=["Y_Int"]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "MatMulInteger" in c


def test_missing_quantization_and_spatial():
    """Test missing quantization and spatial."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test_miss")
    g.tensors["X_Q"] = Tensor("X_Q", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["Scale"] = Tensor("Scale", shape=(1,), dtype=DType.FLOAT32)
    g.tensors["Y_Q"] = Tensor("Y_Q", shape=(1, 5), dtype=DType.INT8)
    g.nodes.append(Node("QuantizeLinear", inputs=["X_Q", "Scale"], outputs=["Y_Q"]))
    g.tensors["X_DQ"] = Tensor("X_DQ", shape=(1, 5), dtype=DType.INT8)
    g.tensors["Y_DQ"] = Tensor("Y_DQ", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("DequantizeLinear", inputs=["X_DQ", "Scale"], outputs=["Y_DQ"]))
    g.tensors["X_CT"] = Tensor("X_CT", shape=(1, 3, 10, 10), dtype=DType.FLOAT32)
    g.tensors["W_CT"] = Tensor("W_CT", shape=(3, 6, 3, 3), dtype=DType.FLOAT32)
    g.tensors["B_CT"] = Tensor("B_CT", shape=(6,), dtype=DType.FLOAT32)
    g.tensors["Y_CT"] = Tensor("Y_CT", shape=(1, 6, 12, 12), dtype=DType.FLOAT32)
    g.nodes.append(Node("ConvTranspose", inputs=["X_CT", "W_CT", "B_CT"], outputs=["Y_CT"]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()
    assert "Add Bias" in c


def test_more_ops_and_broadcasting():
    """Test more ops and broadcasting."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Graph, Node, Tensor

    g = Graph("test_more")
    g.tensors["X_S"] = Tensor("X_S", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["Y_Sign"] = Tensor("Y_Sign", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["Y_Neg"] = Tensor("Y_Neg", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("Sign", inputs=["X_S"], outputs=["Y_Sign"]))
    g.nodes.append(Node("Neg", inputs=["X_S"], outputs=["Y_Neg"]))
    g.tensors["X_B1"] = Tensor("X_B1", shape=(2, 3, 4), dtype=DType.FLOAT32)
    g.tensors["X_B2"] = Tensor("X_B2", shape=(3, 1), dtype=DType.FLOAT32)
    g.tensors["X_B3"] = Tensor("X_B3", shape=(1,), dtype=DType.FLOAT32)
    g.tensors["Y_B1"] = Tensor("Y_B1", shape=(2, 3, 4), dtype=DType.FLOAT32)
    g.tensors["Y_B2"] = Tensor("Y_B2", shape=(2, 3, 4), dtype=DType.FLOAT32)
    g.nodes.append(Node("Add", inputs=["X_B1", "X_B2"], outputs=["Y_B1"]))
    g.nodes.append(Node("Add", inputs=["X_B1", "X_B3"], outputs=["Y_B2"]))
    g.tensors["X_B4"] = Tensor("X_B4", shape=(1, 3, 1), dtype=DType.FLOAT32)
    g.tensors["Y_B3"] = Tensor("Y_B3", shape=(2, 3, 4), dtype=DType.FLOAT32)
    g.nodes.append(Node("Add", inputs=["X_B1", "X_B4"], outputs=["Y_B3"]))
    g.tensors["M_A"] = Tensor("M_A", shape=(4, 5), dtype=DType.FLOAT32)
    g.tensors["M_B"] = Tensor("M_B", shape=(6, 5), dtype=DType.FLOAT32)
    g.tensors["M_C"] = Tensor("M_C", shape=(4,), dtype=DType.FLOAT32)
    g.tensors["M_Y"] = Tensor("M_Y", shape=(4, 6), dtype=DType.FLOAT32)
    g.nodes.append(
        Node(
            "Gemm",
            inputs=["M_A", "M_B", "M_C"],
            outputs=["M_Y"],
            attributes={
                "alpha": Attribute("alpha", value=0.5),
                "beta": Attribute("beta", value=0.5),
                "transB": Attribute("transB", value=1),
            },
        )
    )
    g.tensors["S_A"] = Tensor("S_A", shape=("N", 5), dtype=DType.FLOAT32)
    g.tensors["S_B"] = Tensor("S_B", shape=(5, "M"), dtype=DType.FLOAT32)
    g.tensors["S_Y"] = Tensor("S_Y", shape=("N", "M"), dtype=DType.FLOAT32)
    g.nodes.append(Node("MatMul", inputs=["S_A", "S_B"], outputs=["S_Y"]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()


def test_missing_math_and_targets():
    """Test missing math and targets."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test_math")
    g.tensors["X"] = Tensor("X", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["X2"] = Tensor("X2", shape=(1, 5), dtype=DType.FLOAT32)
    ops = ["Sub", "Mul", "Sqrt", "Pow", "Sin", "Cos", "Tan", "Abs", "Ceil", "Floor", "Round", "Log"]
    for op in ops:
        out = f"Y_{op}"
        g.tensors[out] = Tensor(out, shape=(1, 5), dtype=DType.FLOAT32)
        if op in ["Sub", "Mul", "Pow"]:
            g.nodes.append(Node(op, inputs=["X", "X2"], outputs=[out]))
        else:
            g.nodes.append(Node(op, inputs=["X"], outputs=[out]))
    g.tensors["UINT8_CONST"] = Constant(
        "UINT8_CONST", shape=(2,), dtype=DType.UINT8, values=b"\x01\x02"
    )
    comp_ard = C89Compiler(g, emit_cpp=True, target="arduino")
    (h, c) = comp_ard.generate()
    assert "PROGMEM" in c
    assert 'extern "C"' in c
    comp_bm = C89Compiler(g, emit_cpp=True, target="baremetal")
    (h2, c2) = comp_bm.generate()
    assert "section" in c2
    assert "Sub" in c2
    assert "Mul" in c2


def test_missing_quant_and_pool():
    """Test missing quant and pool."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test_miss")
    g.tensors["X2D"] = Tensor("X2D", shape=(1, 3, 10, 10), dtype=DType.FLOAT32)
    g.tensors["Y_Avg2D"] = Tensor("Y_Avg2D", shape=(1, 3, 5, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("AveragePool", inputs=["X2D"], outputs=["Y_Avg2D"]))
    g.tensors["X2D_Int8"] = Tensor("X2D_Int8", shape=(1, 3, 10, 10), dtype=DType.INT8)
    g.tensors["Scale"] = Constant(
        "Scale", shape=(1,), dtype=DType.FLOAT32, values=struct.pack("<f", 0.1)
    )
    g.tensors["ZP"] = Constant("ZP", shape=(1,), dtype=DType.INT8, values=b"\x00")
    g.tensors["W_Int8"] = Constant(
        "W_Int8", shape=(6, 3, 3, 3), dtype=DType.INT8, values=b"\x01" * 6 * 3 * 3 * 3
    )
    g.tensors["Y_Int8"] = Tensor("Y_Int8", shape=(1, 6, 8, 8), dtype=DType.INT8)
    g.nodes.append(
        Node(
            "QLinearConv",
            inputs=["X2D_Int8", "Scale", "ZP", "W_Int8", "Scale", "ZP", "Scale", "ZP"],
            outputs=["Y_Int8"],
        )
    )
    g.tensors["X_Q"] = Tensor("X_Q", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["Y_Q"] = Tensor("Y_Q", shape=(1, 5), dtype=DType.INT8)
    g.nodes.append(Node("QuantizeLinear", inputs=["X_Q", "Scale"], outputs=["Y_Q"]))
    compiler = C89Compiler(g)
    (h, c) = compiler.generate()


def test_intrinsics():
    """Test intrinsics."""
    from onnx9000.c_compiler.ast_builder import C89Builder
    from onnx9000.c_compiler.intrinsics import (
        apply_simd_unroll,
        emit_cmsis_nn_qlinear_conv,
        emit_cmsis_nn_qlinear_matmul,
    )
    from onnx9000.core.ir import Node

    b = C89Builder()
    n = Node("QLinearMatMul", inputs=[], outputs=[])
    emit_cmsis_nn_qlinear_matmul(b, n, "in1", "in2", "out", 2, 3, 4, 1, 2, 3, 0.5)
    n2 = Node("QLinearConv", inputs=[], outputs=[])
    emit_cmsis_nn_qlinear_conv(
        b,
        n2,
        "in1",
        "w",
        "bias",
        "out",
        10,
        10,
        3,
        3,
        8,
        8,
        3,
        6,
        1,
        2,
        3,
        0.5,
        [1, 1],
        [0, 0],
        [1, 1],
    )
    apply_simd_unroll(b, "desktop")
    apply_simd_unroll(b, "riscv-v")
    apply_simd_unroll(b, "unknown")
    c = b.get_code()
    assert "arm_fully_connected_s8" in c
    assert "arm_convolve_s8" in c
    assert "#pragma omp parallel for" in c
    assert "#pragma GCC unroll 8" in c


def test_target_cmsis():
    """Test target cmsis."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test")
    g.tensors["Q_A"] = Tensor("Q_A", shape=(1, 2, 2), dtype=DType.INT8)
    g.tensors["Q_B"] = Tensor("Q_B", shape=(1, 2, 2), dtype=DType.INT8)
    g.tensors["S"] = Tensor("S", shape=(1,), dtype=DType.FLOAT32)
    g.tensors["ZP"] = Tensor("ZP", shape=(1,), dtype=DType.INT8)
    g.tensors["Q_C"] = Tensor("Q_C", shape=(1, 2, 2), dtype=DType.INT8)
    g.nodes.append(
        Node(
            "QLinearMatMul", inputs=["Q_A", "S", "ZP", "Q_B", "S", "ZP", "S", "ZP"], outputs=["Q_C"]
        )
    )
    g.tensors["QC_X"] = Tensor("QC_X", shape=(1, 3, 10, 10), dtype=DType.UINT8)
    g.tensors["QC_W"] = Tensor("QC_W", shape=(6, 3, 3, 3), dtype=DType.UINT8)
    g.tensors["QC_Y"] = Tensor("QC_Y", shape=(1, 6, 8, 8), dtype=DType.UINT8)
    g.nodes.append(
        Node(
            "QLinearConv",
            inputs=["QC_X", "S", "ZP", "QC_W", "S", "ZP", "S", "ZP"],
            outputs=["QC_Y"],
        )
    )
    comp = C89Compiler(g, target="cmsis-nn")
    (h, c) = comp.generate()


def test_target_esp():
    """Test target esp."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test")
    g.tensors["Q_A"] = Tensor("Q_A", shape=(1, 2, 2), dtype=DType.INT8)
    g.tensors["Q_B"] = Tensor("Q_B", shape=(1, 2, 2), dtype=DType.INT8)
    g.tensors["S"] = Tensor("S", shape=(1,), dtype=DType.FLOAT32)
    g.tensors["ZP"] = Tensor("ZP", shape=(1,), dtype=DType.INT8)
    g.tensors["Q_C"] = Tensor("Q_C", shape=(1, 2, 2), dtype=DType.INT8)
    g.nodes.append(
        Node(
            "QLinearMatMul", inputs=["Q_A", "S", "ZP", "Q_B", "S", "ZP", "S", "ZP"], outputs=["Q_C"]
        )
    )
    g.tensors["QC_X"] = Tensor("QC_X", shape=(1, 3, 10, 10), dtype=DType.UINT8)
    g.tensors["QC_W"] = Tensor("QC_W", shape=(6, 3, 3, 3), dtype=DType.UINT8)
    g.tensors["QC_Y"] = Tensor("QC_Y", shape=(1, 6, 8, 8), dtype=DType.UINT8)
    g.nodes.append(
        Node(
            "QLinearConv",
            inputs=["QC_X", "S", "ZP", "QC_W", "S", "ZP", "S", "ZP"],
            outputs=["QC_Y"],
        )
    )
    comp = C89Compiler(g, target="esp-nn")
    (h, c) = comp.generate()


def test_phase_13_nlp_vision():
    """Test phase 13 nlp vision."""
    from onnx9000.c_compiler.ast_builder import C89Builder
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.c_compiler.nlp import emit_bpe_tokenizer
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Graph, Node, Tensor

    b = C89Builder()
    emit_bpe_tokenizer(b, {"a": 1, "b": 2})
    assert "bpe_vocab" in b.get_code()
    g = Graph("test_p13")
    g.tensors["X"] = Tensor("X", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["Y_NMS"] = Tensor("Y_NMS", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("NonMaxSuppression", inputs=["X", "X", "X", "X", "X"], outputs=["Y_NMS"]))
    g.tensors["Y_Resize"] = Tensor("Y_Resize", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(
        Node(
            "Resize",
            inputs=["X"],
            outputs=["Y_Resize"],
            attributes={"mode": Attribute("mode", b"linear")},
        )
    )
    g.tensors["Y_TopK_Val"] = Tensor("Y_TopK_Val", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["Y_TopK_Idx"] = Tensor("Y_TopK_Idx", shape=(1, 5), dtype=DType.INT64)
    g.nodes.append(Node("TopK", inputs=["X", "X"], outputs=["Y_TopK_Val", "Y_TopK_Idx"]))
    g.tensors["Y_Unique"] = Tensor("Y_Unique", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("Unique", inputs=["X"], outputs=["Y_Unique"]))
    g.tensors["Y_LSTM"] = Tensor("Y_LSTM", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("LSTM", inputs=["X", "X", "X"], outputs=["Y_LSTM"]))
    g.tensors["Y_Attn"] = Tensor("Y_Attn", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("Attention", inputs=["X", "X", "X", "X"], outputs=["Y_Attn"]))
    comp = C89Compiler(g)
    comp.generate()


def test_avx2():
    """Test avx2."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.ir import Graph

    g = Graph("test")
    c = C89Compiler(g, target="avx2")
    (h, cc) = c.generate()
    assert "__AVX2__" in cc


def test_missing_nlp_and_rnn():
    """Test missing nlp and rnn."""
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Graph, Node, Tensor

    g = Graph("test")
    g.tensors["X"] = Tensor("X", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["Y_TopK_Val"] = Tensor("Y_TopK_Val", shape=(1, 5), dtype=DType.FLOAT32)
    g.tensors["Y_TopK_Idx"] = Tensor("Y_TopK_Idx", shape=(1, 5), dtype=DType.INT64)
    g.nodes.append(Node("TopK", inputs=["X", "X"], outputs=["Y_TopK_Val", "Y_TopK_Idx"]))
    g.tensors["Y_Unique"] = Tensor("Y_Unique", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("Unique", inputs=["X"], outputs=["Y_Unique"]))
    g.tensors["Y_NMS"] = Tensor("Y_NMS", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("NonMaxSuppression", inputs=["X", "X", "X", "X", "X"], outputs=["Y_NMS"]))
    g.tensors["Y_NMS2"] = Tensor("Y_NMS2", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("NonMaxSuppression", inputs=["X", "X", "X", "X"], outputs=["Y_NMS2"]))
    g.tensors["Y_Resize"] = Tensor("Y_Resize", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(
        Node(
            "Resize",
            inputs=["X"],
            outputs=["Y_Resize"],
            attributes={"mode": Attribute("mode", b"linear")},
        )
    )
    g.tensors["Y_Resize2"] = Tensor("Y_Resize2", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(
        Node(
            "Resize",
            inputs=["X"],
            outputs=["Y_Resize2"],
            attributes={"mode": Attribute("mode", "nearest")},
        )
    )
    g.tensors["Y_LSTM"] = Tensor("Y_LSTM", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("LSTM", inputs=["X", "X", "X"], outputs=["Y_LSTM"]))
    g.tensors["Y_Attn"] = Tensor("Y_Attn", shape=(1, 5), dtype=DType.FLOAT32)
    g.nodes.append(Node("Attention", inputs=["X", "X", "X", "X"], outputs=["Y_Attn"]))
    c = C89Compiler(g)
    c.generate()
