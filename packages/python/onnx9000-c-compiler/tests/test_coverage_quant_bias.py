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
