"""Unit tests for IRNode, IRGraph, OracleVerifier, check_tolerance, and bisect_dag."""

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


def test_ir_node_and_graph() -> None:
    """Verify IRNode and IRGraph construction."""
    node = IRNode("conv_0", "Conv")
    assert node.name == "conv_0"
    assert node.op_type == "Conv"

    graph = IRGraph([node])
    assert len(graph.nodes) == 1
    assert graph.nodes[0] is node


def test_oracle_verifier_and_tolerance() -> None:
    """Verify OracleVerifier, check_tolerance, reset_environment, and bisect_dag."""
    reset_environment(123)
    verifier = OracleVerifier(lambda x: x, dtype="FP32")
    assert verifier.verify((2, 2)) is True

    passed, diff = check_tolerance([1.0, 2.0], [1.0, 2.0], "FP32")
    assert passed is True
    assert diff == 0.0

    with pytest.raises(ValueError, match="Unsupported dtype"):
        check_tolerance([1.0], [1.0], "INVALID")

    n0 = IRNode("n0", "Relu")
    g = IRGraph([n0])
    assert bisect_dag(g, lambda _: [1.0], lambda _: [1.0]) is None
    assert bisect_dag(g, lambda _: [1.0], lambda _: [10.0]) is n0
