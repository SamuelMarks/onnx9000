"""Unit tests for OperatorRegistry in `onnx9000.core.registry`."""

from onnx9000.core.registry import OperatorRegistry, register_op


def test_operator_registry() -> None:
    """Verify OperatorRegistry registration and lookup."""
    reg = OperatorRegistry()

    @reg.register_op("custom", "TestOp")
    def _op() -> str:
        """Execute op operation.

        Returns:
            Result of op (str).
        """
        return "ok"

    assert reg.get_op("custom", "TestOp")() == "ok"

    @register_op("custom", "GlobalTestOp")
    def _gop() -> str:
        """Execute gop operation.

        Returns:
            Result of gop (str).
        """
        return "g_ok"

    assert _gop() == "g_ok"
