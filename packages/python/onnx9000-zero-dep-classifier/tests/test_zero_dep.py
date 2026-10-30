"""Unit tests for onnx9000_zero_dep_classifier re-exports."""

from onnx9000_zero_dep_classifier import ZeroDepClassifier


def test_zero_dep() -> None:
    """Verify ZeroDepClassifier.process via onnx9000_zero_dep_classifier."""
    z = ZeroDepClassifier()
    assert z.process("test") == "Zero Dep Classifier processed test"
