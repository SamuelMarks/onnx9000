from onnx9000.extensions.custom.math_ops import matmul


def test_matmul_cov():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    C = matmul(A, B)
    assert C[0][0] == 19.0
