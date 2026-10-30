"""Backward-compatible re-export module for `onnx9000_zero_dep_classifier`."""

from onnx9000.zero_dep_classifier import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ClassificationPrediction,
    ZeroDepClassifier,
    normalize_rgb_pixel,
    softmax_probabilities,
)

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ClassificationPrediction",
    "ZeroDepClassifier",
    "normalize_rgb_pixel",
    "softmax_probabilities",
]
