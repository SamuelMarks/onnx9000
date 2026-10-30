"""Tests for pure-Python numerical verification and DAG bisection utilities."""

from __future__ import annotations

import pytest
from onnx9000.core.verification import (
    IRGraph,
    IRNode,
    OracleVerifier,
    bisect_dag,
    check_tolerance,
    reset_environment,
)


class _ListBackedTensor:
    """Mock tensor implementing FloatTensorProtocol."""

    def __init__(self, vals: list[float] | float) -> None:
        """Initialize with values."""
        self._vals = vals

    def tolist(self) -> list[float] | float:
        """Return stored values."""
        return self._vals


def test_check_tolerance_precisions() -> None:
    """Verify check_tolerance across FP32, FP16, BF16, INT8, Q4_0, and empty inputs."""
    assert check_tolerance([1.0, 2.0], [1.000001, 2.000001], "FP32")[0] is True
    assert check_tolerance([1.0, 2.0], [1.001, 2.001], "FP32")[0] is False

    assert check_tolerance([1.0, 2.0], [1.0005, 2.0005], "FP16")[0] is True
    assert check_tolerance([1.0, 2.0], [1.05, 2.05], "FP16")[0] is False

    assert check_tolerance([1.0, 2.0], [1.005, 2.005], "BF16")[0] is True
    assert check_tolerance([1.0, 2.0], [1.1, 2.1], "BF16")[0] is False

    assert check_tolerance([1.0, 0.0], [0.99, 0.1], "INT8")[0] is True
    assert check_tolerance([1.0, 0.0], [0.0, 1.0], "INT8")[0] is False
    assert check_tolerance([0.0, 0.0], [0.0, 0.0], "INT8") == (True, 1.0)
    assert check_tolerance([1.0, 0.0], [0.0, 0.0], "Q4_0") == (False, 0.0)

    assert check_tolerance([], [], "FP32") == (True, 0.0)
    assert check_tolerance([[1.0, 2.0]], [[1.0, 2.0]], "FP32")[0] is True
    assert (
        check_tolerance(_ListBackedTensor([1.0, 2.0]), _ListBackedTensor([1.0, 2.0]), "FP32")[0]
        is True
    )
    assert check_tolerance(_ListBackedTensor(1.0), _ListBackedTensor(1.0), "FP32")[0] is True

    with pytest.raises(ValueError, match="Unsupported dtype"):
        check_tolerance([1.0], [1.0], "UNKNOWN")


def test_reset_environment_and_oracle_verifier() -> None:
    """Verify deterministic seeding, OracleVerifier pass/fail, and bisect_dag."""
    reset_environment(42)
    v1 = OracleVerifier(lambda x: [v * 2.0 for v in x], dtype="FP32")
    in1 = v1.generate_inputs((2, 3))
    reset_environment(42)
    in2 = v1.generate_inputs((2, 3))
    assert in1 == in2
    assert v1.verify((2, 3)) is True

    class DivergingVerifier(OracleVerifier):
        """Verifier whose target diverges from the oracle."""

        def run_target(self, inputs: list[float], artifacts: None) -> list[float]:
            """Return shifted outputs."""
            _ = artifacts
            return [v + 10.0 for v in inputs]

    v_fail = DivergingVerifier(lambda x: x, dtype="FP32")
    assert v_fail.verify((2, 2)) is False

    nodes = [IRNode("n1", "Add"), IRNode("n2", "Mul")]
    graph = IRGraph(nodes)
    assert bisect_dag(graph, lambda _: [1.0, 2.0], lambda _: [1.0, 2.0]) is None
    failing = bisect_dag(
        graph,
        lambda _: [1.0, 2.0],
        lambda idx: [1.0, 2.0] if idx == 0 else [5.0, 5.0],
    )
    assert failing is not None
    assert failing.name == "n2"
