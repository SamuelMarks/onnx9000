"""Tests for quantized convolution with bias in the C compiler."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.c_compiler.quantization import generate_qlinear_conv
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Node, Tensor


def test_qlinear_conv_bias():
    """Test QLinearConv with bias."""
    b = C89Builder()
    n = Node("QLinearConv", ["X", "xS", "xZ", "W", "wS", "wZ", "yS", "yZ", "B"], ["Y"])
    # 9 inputs implies bias is included
    tX = Tensor("X", [1, 1, 3, 3], DType.UINT8, data=b"\\x00" * 32)
    xS = Tensor("xS", [1], DType.FLOAT32, data=b"\\x00" * 32)
    xZ = Tensor("xZ", [1], DType.UINT8, data=b"\\x00" * 32)
    tW = Tensor("W", [1, 1, 3, 3], DType.UINT8, data=b"\\x00" * 32)
    wS = Tensor("wS", [1], DType.FLOAT32, data=b"\\x00" * 32)
    wZ = Tensor("wZ", [1], DType.UINT8, data=b"\\x00" * 32)
    yS = Tensor("yS", [1], DType.FLOAT32, data=b"\\x00" * 32)
    yZ = Tensor("yZ", [1], DType.UINT8, data=b"\\x00" * 32)
    tB = Tensor("B", [1], DType.INT32, data=b"\\x00" * 32)

    tY = Tensor("Y", [1, 1, 3, 3], DType.UINT8, data=b"\\x00" * 32)

    generate_qlinear_conv(
        b,
        n,
        tY,
        tX,
        xS,
        xZ,
        tW,
        wS,
        wZ,
        yS,
        yZ,
        "X",
        "xS",
        "xZ",
        "W",
        "wS",
        "wZ",
        "yS",
        "yZ",
        "Y",
        tB,
        "B",
    )


def test_dequantize_linear_miss():
    """Docstring for D103."""
    from onnx9000.c_compiler.ast_builder import C89Builder
    from onnx9000.c_compiler.quantization import generate_dequantize_linear
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Node, Tensor

    b = C89Builder()
    n = Node("DequantizeLinear", ["X", "xS", "xZ"], ["Y"])
    tX = Tensor("X", [1, 3], DType.INT8, data=b"\x00" * 32)
    xS = Tensor("xS", [1], DType.FLOAT32, data=b"\x00" * 32)
    Tensor("xZ", [1], DType.INT8, data=b"\x00" * 32)
    tY = Tensor("Y", [1, 3], DType.FLOAT32, data=b"\x00" * 32)

    # Missing args (missing inputs)
    n.inputs = ["X"]
    generate_dequantize_linear(b, n, tY, tX, xS, None, "X", "xS", "", "Y")
    code = b.get_code()
    assert "DequantizeLinear" in code


def test_quantize_linear_miss():
    """Docstring for D103."""
    from onnx9000.c_compiler.ast_builder import C89Builder
    from onnx9000.c_compiler.quantization import generate_quantize_linear
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Node, Tensor

    b = C89Builder()
    n = Node("QuantizeLinear", ["X", "xS", "xZ"], ["Y"])
    tX = Tensor("X", [1, 3], DType.FLOAT32, data=b"\x00" * 32)
    xS = Tensor("xS", [1], DType.FLOAT32, data=b"\x00" * 32)
    Tensor("xZ", [1], DType.UINT8, data=b"\x00" * 32)
    tY = Tensor("Y", [1, 3], DType.UINT8, data=b"\x00" * 32)

    # missing args
    n.inputs = ["X"]
    generate_quantize_linear(b, n, tY, tX, xS, None, "X", "xS", "", "Y")
    code = b.get_code()
    assert "QuantizeLinear" in code


def test_qlinear_matmul_miss():
    """Docstring for D103."""
    from onnx9000.c_compiler.ast_builder import C89Builder
    from onnx9000.c_compiler.quantization import generate_qlinear_matmul
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Node, Tensor

    b = C89Builder()
    n = Node("QLinearMatMul", ["X", "xS", "xZ", "W", "wS", "wZ", "yS", "yZ"], ["Y"])
    tX = Tensor("X", [1, 3], DType.UINT8, data=b"\x00" * 32)
    xS = Tensor("xS", [1], DType.FLOAT32, data=b"\x00" * 32)
    Tensor("xZ", [1], DType.UINT8, data=b"\x00" * 32)
    tW = Tensor("W", [1, 3], DType.UINT8, data=b"\x00" * 32)
    wS = Tensor("wS", [1], DType.FLOAT32, data=b"\x00" * 32)
    wZ = Tensor("wZ", [1], DType.UINT8, data=b"\x00" * 32)
    yS = Tensor("yS", [1], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)
    yZ = Tensor("yZ", [1], DType.UINT8, data=b"\x00" * 32)
    tY = Tensor("Y", [1, 3], DType.UINT8, data=b"\x00" * 32)

    # Missing args handling
    n.inputs = ["X"]
    generate_qlinear_matmul(
        b, n, tY, tX, xS, None, tW, wS, wZ, yS, yZ, "X", "xS", "", "W", "wS", "wZ", "yS", "yZ", "Y"
    )
    code = b.get_code()
    assert "QLinearMatMul" in code


def test_qlinear_matmul_miss_2():
    """Docstring for D103."""
    from onnx9000.c_compiler.ast_builder import C89Builder
    from onnx9000.c_compiler.quantization import generate_qlinear_matmul
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Node, Tensor

    b = C89Builder()
    n = Node("QLinearMatMul", ["X", "xS", "xZ", "W", "wS", "wZ", "yS", "yZ"], ["Y"])
    tX = Tensor("X", [1, 3], DType.UINT8, data=b"\x00" * 32)
    xS = Tensor("xS", [1], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)
    xZ = Tensor("xZ", [1], DType.UINT8, data=b"\x00" * 32)
    tW = Tensor("W", [1, 3], DType.UINT8, data=b"\x00" * 32)
    wS = Tensor("wS", [1], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)
    wZ = Tensor("wZ", [1], DType.UINT8, data=b"\x00" * 32)
    yS = Tensor("yS", [1], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)
    Tensor("yZ", [1], DType.UINT8, data=b"\x00" * 32)
    tY = Tensor("Y", [1, 3], DType.UINT8, data=b"\x00" * 32)

    # Missing optional arg zp_out_tensor
    generate_qlinear_matmul(
        b, n, tY, tX, xS, xZ, tW, wS, wZ, yS, None, "X", "xS", "xZ", "W", "wS", "wZ", "yS", "", "Y"
    )
    code = b.get_code()
    assert "QLinearMatMul" in code


def test_qlinear_matmul_q4_0():
    """Docstring for D103."""
    from onnx9000.c_compiler.ast_builder import C89Builder
    from onnx9000.c_compiler.quantization import generate_qlinear_matmul
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Node, Tensor

    b = C89Builder()
    n = Node(
        "QLinearMatMul",
        ["X", "xS", "xZ", "W", "wS", "wZ", "yS", "yZ"],
        ["Y"],
        attributes={"use_block_q4_0": Attribute("use_block_q4_0", 1, "INT")},
    )
    tX = Tensor("X", [1, 3], DType.UINT8, data=b"\x00" * 32)
    xS = Tensor("xS", [1], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)
    xZ = Tensor("xZ", [1], DType.UINT8, data=b"\x00" * 32)
    tW = Tensor("W", [1, 3], DType.UINT8, data=b"\x00" * 32)
    wS = Tensor("wS", [1], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)
    wZ = Tensor("wZ", [1], DType.UINT8, data=b"\x00" * 32)
    yS = Tensor("yS", [1], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)
    Tensor("yZ", [1], DType.UINT8, data=b"\x00" * 32)
    tY = Tensor("Y", [1, 3], DType.UINT8, data=b"\x00" * 32)

    generate_qlinear_matmul(
        b, n, tY, tX, xS, xZ, tW, wS, wZ, yS, None, "X", "xS", "xZ", "W", "wS", "wZ", "yS", "", "Y"
    )
    code = b.get_code()
    assert "ggml_vec_dot_q4_0_q8_0" in code
