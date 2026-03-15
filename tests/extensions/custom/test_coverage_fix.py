"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.extensions.custom.math_ops import inverse, svd
from onnx9000.extensions.custom.decoding import sample_top_k_top_p


def test_inverse_singular():
    """Provides semantic functionality and verification."""
    with pytest.raises(ValueError, match="Matrix is singular"):
        inverse([[1.0, 1.0], [1.0, 1.0]])


def test_svd_non_diagonal():
    """Provides semantic functionality and verification."""
    U, S, V = svd([[1.0, 2.0], [3.0, 4.0]])
    assert S[0] > 0


def test_svd_zero():
    """Provides semantic functionality and verification."""
    U, S, V = svd([[0.0, 0.0], [0.0, 0.0]])
    assert S[0] <= 1e-09


def test_sample_top_k_top_p_end():
    """Provides semantic functionality and verification."""
    import random

    random.seed(0)
    import unittest.mock as mock

    with mock.patch("random.random", return_value=1.0):
        res = sample_top_k_top_p([1.0, 2.0], top_k=0, top_p=1.0)
        assert res in [0, 1]
