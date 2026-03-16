"""Module providing core logic and structural definitions."""


def test_opt_stubs() -> None:
    """Tests the test_opt_stubs functionality."""
    from onnx9000.core.ir import Graph
    from onnx9000.toolkit.training.autograd.optimizers import (
        add_gradient_accumulation,
        add_gradient_clipping,
    )

    g = Graph("test")
    add_gradient_accumulation(g, [], 2)
    add_gradient_clipping(g, [], 1.0)
