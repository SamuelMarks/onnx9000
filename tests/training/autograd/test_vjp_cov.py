"""Module providing core logic and structural definitions."""


def test_vjp_base():
    """Provides semantic functionality and verification."""
    from onnx9000.training.autograd.vjp import VJPRule

    class DummyRule(VJPRule):
        """Represents the DummyRule class."""

        def build_backward_nodes(self, n, g):
            """Provides build backward nodes functionality and verification."""
            return super().build_backward_nodes(n, g)

    assert DummyRule().build_backward_nodes(None, []) == ([], [])
