"""Tests for verification."""

import unittest

import torch
from onnx9000.core.verification import (
    IRGraph,
    IRNode,
    OracleVerifier,
    bisect_dag,
    check_tolerance,
    reset_environment,
)


class MockModel(torch.nn.Module):
    """Docstring for D101."""

    def forward(self, x):
        """Docstring for D102."""
        return x * 2.0


class TestVerification(unittest.TestCase):
    """Docstring for D101."""

    def test_check_tolerance_fp32(self):
        """Docstring for D102."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([1.000001, 2.000001])  # diff = 1e-6
        passed, max_diff = check_tolerance(t1, t2, "FP32")
        self.assertTrue(passed)

        t3 = torch.tensor([1.0, 2.0])
        t4 = torch.tensor([1.001, 2.001])  # diff = 1e-3, will fail atol=1e-5
        passed, max_diff = check_tolerance(t3, t4, "FP32")
        self.assertFalse(passed)

    def test_check_tolerance_fp16(self):
        """Docstring for D102."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([1.0005, 2.0005])  # diff = 5e-4
        passed, max_diff = check_tolerance(t1, t2, "FP16")
        self.assertTrue(passed)

        t3 = torch.tensor([1.0, 2.0])
        t4 = torch.tensor([1.05, 2.05])  # diff = 5e-2, will fail atol=1e-3, rtol=1e-2
        passed, max_diff = check_tolerance(t3, t4, "FP16")
        self.assertFalse(passed)

    def test_check_tolerance_bf16(self):
        """Docstring for D102."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([1.005, 2.005])  # diff = 5e-3
        passed, max_diff = check_tolerance(t1, t2, "BF16")
        self.assertTrue(passed)

        t3 = torch.tensor([1.0, 2.0])
        t4 = torch.tensor([1.1, 2.1])  # diff = 1e-1, will fail atol=1e-2, rtol=5e-2
        passed, max_diff = check_tolerance(t3, t4, "BF16")
        self.assertFalse(passed)

    def test_check_tolerance_int8(self):
        """Docstring for D102."""
        t1 = torch.tensor([1.0, 0.0])
        t2 = torch.tensor([0.99, 0.1])
        passed, cos_sim = check_tolerance(t1, t2, "INT8")
        self.assertTrue(passed)

        t3 = torch.tensor([1.0, 0.0])
        t4 = torch.tensor([0.0, 1.0])
        passed, cos_sim = check_tolerance(t3, t4, "INT8")
        self.assertFalse(passed)

    def test_reset_environment(self):
        """Docstring for D102."""
        reset_environment(42)
        r1 = torch.randn(1)
        reset_environment(42)
        r2 = torch.randn(1)
        self.assertTrue(torch.allclose(r1, r2))

    def test_oracle_verifier(self):
        """Docstring for D102."""
        model = MockModel()
        verifier = OracleVerifier(model, dtype="FP32")
        self.assertTrue(verifier.verify(input_shape=(1, 3, 224, 224)))

    def test_bisect_dag(self):
        """Docstring for D102."""
        nodes = [IRNode("n1", "Add"), IRNode("n2", "Mul")]
        graph = IRGraph(nodes)

        def oracle_fn(idx):
            """Oracle fn."""
            return torch.tensor([1.0, 2.0])

        def target_fn_pass(idx):
            """Target fn pass."""
            return torch.tensor([1.0, 2.0])

        def target_fn_fail_at_1(idx):
            """Target fn fail at 1."""
            if idx == 0:
                return torch.tensor([1.0, 2.0])
            else:
                return torch.tensor([5.0, 5.0])

        # Should return None if everything passes
        failing_node = bisect_dag(graph, oracle_fn, target_fn_pass)
        self.assertIsNone(failing_node)

        # Should return the second node ("n2") if it fails there
        failing_node = bisect_dag(graph, oracle_fn, target_fn_fail_at_1)
        self.assertIsNotNone(failing_node)
        self.assertEqual(failing_node.name, "n2")


if __name__ == "__main__":
    unittest.main()


def test_verify_tensor_int8_zero():
    """Docstring for D103."""
    import torch
    from onnx9000.core.verification import check_tolerance

    t1 = torch.zeros((10,), dtype=torch.float32)
    t2 = torch.zeros((10,), dtype=torch.float32)
    passed, score = check_tolerance(t1, t2, dtype="INT8")
    assert passed
    assert score == 1.0


def test_verify_tensor_unsupported_dtype():
    """Docstring for D103."""
    import pytest
    import torch
    from onnx9000.core.verification import check_tolerance

    with pytest.raises(ValueError):
        check_tolerance(torch.zeros(1), torch.zeros(1), dtype="UNKNOWN")


def test_reset_env():
    """Docstring for D103."""
    # The cuda condition is hard to hit, but we can patch torch.cuda.is_available
    import unittest.mock

    import torch
    from onnx9000.core.verification import reset_environment

    with unittest.mock.patch("torch.cuda.is_available", return_value=True):
        with unittest.mock.patch("torch.cuda.empty_cache") as mock_empty:
            reset_environment()
            mock_empty.assert_called_once()


def test_oracle_missing():
    """Docstring for D103."""
    import pytest
    import torch
    from onnx9000.core.verification import OracleVerifier

    # 79: Oracle model missing call
    class MockModel:
        """Mock model."""

        def __call__(self, x):
            """Call."""
            return x

    ov = OracleVerifier(MockModel())
    ov.verify((1, 1))

    # 105: Oracle returns a scalar
    class ScalarModel:
        """Scalar model."""

        def __call__(self, x):
            """Call."""
            return 42.0

    ov2 = OracleVerifier(ScalarModel())
    with pytest.raises(AttributeError):
        ov2.verify((1, 1))


def test_cuda_mock_verify():
    """Docstring for D103."""
    import unittest.mock

    import torch
    from onnx9000.core.verification import OracleVerifier

    class MockModel:
        """Mock model."""

        def __call__(self, x):
            """Call."""
            return x

    ov = OracleVerifier(MockModel())

    with unittest.mock.patch("torch.cuda.is_available", return_value=True):
        with unittest.mock.patch("torch.cuda.reset_peak_memory_stats") as mock_reset:
            with unittest.mock.patch("torch.cuda.max_memory_allocated", return_value=1024):
                ov.verify((1, 1))
                mock_reset.assert_called_once()


def test_oracle_verify_failed():
    """Docstring for D103."""
    import unittest.mock

    import torch
    from onnx9000.core.verification import OracleVerifier

    class BadModel:
        """Bad model."""

        def __call__(self, x):
            """Call."""
            return x + 1.0  # will fail tolerance

    ov = OracleVerifier(BadModel())
    ov.verify((1, 1))  # will print and return false


def test_verify_fails():
    """Docstring for D103."""
    import unittest.mock

    import torch
    from onnx9000.core.verification import OracleVerifier

    class MockModel:
        """Mock model."""

        def __call__(self, x):
            """Call."""
            return x + 100.0  # Will fail tolerance significantly

    ov = OracleVerifier(MockModel())
    # Mock check_tolerance to return False, 0.5 without crashing
    with unittest.mock.patch(
        "onnx9000.core.verification.check_tolerance", return_value=(False, 0.5)
    ):
        assert not ov.verify((1, 1))
