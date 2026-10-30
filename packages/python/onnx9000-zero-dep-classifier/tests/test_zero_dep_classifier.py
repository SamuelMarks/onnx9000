"""Comprehensive unit tests for onnx9000.zero_dep_classifier and onnx9000_zero_dep_classifier."""

from onnx9000.zero_dep_classifier import (
    ZeroDepClassifier,
    normalize_rgb_pixel,
    softmax_probabilities,
)


def test_zero_dep_classifier_pipeline() -> None:
    """Verify ZeroDepClassifier.process(), top_k(), softmax_probabilities(), and normalize_rgb_pixel()."""
    z = ZeroDepClassifier(labels=["cat", "dog"])
    assert z.process("test") == "Zero Dep Classifier processed test"

    assert softmax_probabilities([]) == []
    preds = z.top_k([0.0, 2.0, -1.0], k=2)
    assert len(preds) == 2
    assert preds[0].class_index == 1
    assert preds[0].label == "dog"

    preds_unlabeled = ZeroDepClassifier().top_k([1.0, 3.0], k=2)
    assert preds_unlabeled[0].label == "class_1"

    r_n, g_n, b_n = normalize_rgb_pixel(255, 0, 128)
    assert r_n > 0.0
    assert g_n < 0.0
    assert isinstance(b_n, float)
