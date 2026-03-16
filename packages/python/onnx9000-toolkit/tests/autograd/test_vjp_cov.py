"""Module providing core logic and structural definitions."""


def test_vjp_base() -> None:
    """Tests the test_vjp_base functionality."""
    from onnx9000.toolkit.training.autograd.vjp import VJPRule

    class DummyRule(VJPRule):
        """Class DummyRule implementation."""

        def build_backward_nodes(self, n, g):
            """Tests the build_backward_nodes functionality."""
            return super().build_backward_nodes(n, g)

    assert DummyRule().build_backward_nodes(None, []) == ([], [])
