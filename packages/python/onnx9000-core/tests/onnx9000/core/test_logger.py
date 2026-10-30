"""Unit tests for logger factory in `onnx9000.core.logger`."""

from onnx9000.core.logger import get_logger


def test_get_logger() -> None:
    """Verify get_logger returns a configured Logger instance."""
    log = get_logger("onnx9000.test")
    assert log.name == "onnx9000.test"
    assert len(log.handlers) >= 1
