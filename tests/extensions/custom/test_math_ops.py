"""Module providing core logic and structural definitions."""

import pytest
import math
from onnx9000.extensions.custom.math_ops import inverse, svd, einsum


def test_inverse():
    """Provides semantic functionality and verification."""
    mat = [[4, 7], [2, 6]]
    inv = inverse(mat)
    assert math.isclose(inv[0][0], 0.6, abs_tol=1e-05)
    assert math.isclose(inv[0][1], -0.7, abs_tol=1e-05)
    assert math.isclose(inv[1][0], -0.2, abs_tol=1e-05)
    assert math.isclose(inv[1][1], 0.4, abs_tol=1e-05)


def test_inverse_singular():
    """Provides semantic functionality and verification."""
    mat = [[1, 1], [1, 1]]
    with pytest.raises(ValueError):
        inverse(mat)


def test_svd():
    """Provides semantic functionality and verification."""
    mat = [[1.0, 0.0], [0.0, 1.0]]
    u, s, vt = svd(mat)
    assert math.isclose(s[0], 1.0, abs_tol=1e-05)
    assert math.isclose(s[1], 1.0, abs_tol=1e-05)


def test_einsum_matmul():
    """Provides semantic functionality and verification."""
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = einsum("ij,jk->ik", A, B)
    assert C == [[19, 22], [43, 50]]


def test_einsum_trace():
    """Provides semantic functionality and verification."""
    A = [[1, 2], [3, 4]]
    res = einsum("ii->", A)
    assert res == 5


def test_einsum_transpose():
    """Provides semantic functionality and verification."""
    A = [[1, 2], [3, 4]]
    C = einsum("ij->ji", A)
    assert C == [[1, 3], [2, 4]]


def test_einsum_sum():
    """Provides semantic functionality and verification."""
    A = [1, 2, 3]
    res = einsum("i->", A)
    assert res == 6


def test_einsum_errors():
    """Provides semantic functionality and verification."""
    A = [[1]]
    B = [[2]]
    with pytest.raises(ValueError):
        einsum("ij,jk->ik", A)
    with pytest.raises(ValueError):
        einsum("ijk->", A)
    with pytest.raises(ValueError):
        einsum("i->k", A)
