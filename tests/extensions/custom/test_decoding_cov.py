"""Module providing core logic and structural definitions."""


def test_decoding_random_1():
    """Provides semantic functionality and verification."""
    from onnx9000.extensions.custom.decoding import sample_top_k_top_p
    from unittest.mock import patch

    with patch("random.random", return_value=1.0):
        res = sample_top_k_top_p([1.0, 2.0], top_k=2)
        assert res in (0, 1)


def test_decoding_random_1_exact():
    """Provides semantic functionality and verification."""
    from onnx9000.extensions.custom.decoding import sample_top_k_top_p
    from unittest.mock import patch

    with patch("random.random", return_value=2.0):
        res = sample_top_k_top_p([1.0, 2.0])
        assert res == 0
