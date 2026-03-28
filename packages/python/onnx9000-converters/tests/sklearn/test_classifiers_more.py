"""Tests the classifiers more module functionality."""

import pytest
from onnx9000.converters.sklearn.linear import _convert_linear_classifier
from onnx9000.core.ir import Graph


class MockEstimator:
    """Represents the Mock Estimator class."""

    def __init__(self):
        """Initialize the instance."""
        import numpy as np

        self.coef_ = np.array([[1.0, 2.0]])
        self.intercept_ = np.array([0.5])
        self.classes_ = ["A", "B"]


def test_linear_classifier_strings():
    """Tests the linear classifier strings functionality."""
    g = Graph("g")
    _convert_linear_classifier(MockEstimator(), ["input"], g)
    assert "classlabels_strings" in g.nodes[-2].attrs


class MockEstimatorInts:
    """Represents the Mock Estimator Ints class."""

    def __init__(self):
        """Initialize the instance."""
        import numpy as np

        self.coef_ = np.array([[1.0, 2.0]])
        self.intercept_ = np.array([0.5])
        self.classes_ = [0, 1]


def test_linear_classifier_ints():
    """Tests the linear classifier ints functionality."""
    g = Graph("g")
    _convert_linear_classifier(MockEstimatorInts(), ["input"], g)
    assert "classlabels_int64s" in g.nodes[-2].attrs
