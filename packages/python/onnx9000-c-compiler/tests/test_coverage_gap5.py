import pytest
from onnx9000.c_compiler.operations import get_strides, resolve_broadcast_indices
from onnx9000.c_compiler.ast_builder import C89Builder


def test_operations_gemm():
    assert resolve_broadcast_indices([1], [1]) == "0"
    b = C89Builder()

    from onnx9000.c_compiler.operations import generate_matmul
    from onnx9000.core.ir import Node, Tensor
    from onnx9000.core.dtypes import DType

    n = Node("Gemm", ["A", "B"], ["C"], attributes={"alpha": 2.0})
    tC = Tensor("C", [2, 2], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)
    tA = Tensor("A", [2, 2], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)
    tB = Tensor("B", [2, 2], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)

    generate_matmul(b, n, tC, tA, tB, "A", "B", "C", False, False, "C", True, 2.0, 1.0)


def test_quant_matmul_nd():
    from onnx9000.c_compiler.quantization import generate_qlinear_matmul
    from onnx9000.core.ir import Node, Tensor
    from onnx9000.core.dtypes import DType

    b = C89Builder()
    n = Node("QLinearMatMul", ["A", "AS", "AZ", "B", "BS", "BZ", "YS", "YZ"], ["Y"])

    tA = Tensor("A", [2, 2, 2, 2], DType.UINT8, data=b"\x00\x00\x80\x3f" * 8)
    sA = Tensor("AS", [1], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)
    zA = Tensor("AZ", [1], DType.UINT8, data=b"\x00\x00\x80\x3f" * 8)

    tB = Tensor("B", [2, 2, 2, 2], DType.UINT8, data=b"\x00\x00\x80\x3f" * 8)
    sB = Tensor("BS", [1], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)
    zB = Tensor("BZ", [1], DType.UINT8, data=b"\x00\x00\x80\x3f" * 8)

    sY = Tensor("YS", [1], DType.FLOAT32, data=b"\x00\x00\x80\x3f" * 8)
    zY = Tensor("YZ", [1], DType.UINT8, data=b"\x00\x00\x80\x3f" * 8)

    tY = Tensor("Y", [2, 2, 2, 2], DType.UINT8, data=b"\x00\x00\x80\x3f" * 8)

    generate_qlinear_matmul(
        b, n, tY, tA, sA, zA, tB, sB, zB, sY, zY, "A", "sA", "zA", "B", "sB", "zB", "sY", "zY", "Y"
    )


def test_subnormal_float():
    from onnx9000.c_compiler.data_unpacker import unpack_bytes_to_str
    from onnx9000.core.dtypes import DType
    import struct

    data = struct.pack("<f", 1e-35)
    result = unpack_bytes_to_str(data, DType.FLOAT32)
    assert result == "0.0f"
    data64 = struct.pack("<d", 1e-35)
    result64 = unpack_bytes_to_str(data64, DType.FLOAT64)
    assert result64 == "0.0f"
