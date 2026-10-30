"""Unit tests for onnx9000.optimum.optimize and onnx9000_optimum.optimize."""

from unittest.mock import MagicMock, patch

from onnx9000.optimum.optimize import optimize_model


def test_optimize_model_stat_oserror() -> None:
    """Verify optimize_model returns False when os.path.getsize raises OSError."""
    with (
        patch("onnx9000.core.parser.core.load", return_value=MagicMock()),
        patch("onnx9000.core.serializer.save"),
        patch("onnx9000.optimizer.simplifier.api.simplify", return_value=MagicMock()),
        patch("os.path.getsize", side_effect=OSError("missing file")),
    ):
        assert optimize_model("missing.onnx", level="O1") is False
