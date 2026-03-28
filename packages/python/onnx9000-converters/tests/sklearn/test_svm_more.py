"""Tests the svm more module functionality."""

import numpy as np
import pytest
from onnx9000.converters.sklearn.svm import _convert_svm_classifier
from onnx9000.core.ir import Graph


class MockSVC:
    """Represents the Mock S V C class."""

    def __init__(self):
        """Initialize the instance."""
        self.n_support_ = np.array([1, 1])
        self.support_vectors_ = np.array([[1.0]])
        self.dual_coef_ = np.array([[0.5, -0.5]])
        self.intercept_ = np.array([0.1])
        self.probA_ = np.array([])
        self.probB_ = np.array([])
        self.classes_ = np.array([0, 1])
        self.kernel = "rbf"
        self._gamma = 0.5
        self.coef0 = 0.0
        self.degree = 3


def test_svm_classifier_ints():
    """Tests the svm classifier ints functionality."""
    g = Graph("g")
    _convert_svm_classifier(MockSVC(), ["input"], g)
    assert "classlabels_ints" in g.nodes[-2].attrs


class MockSVCStrings:
    """Represents the Mock S V C Strings class."""

    def __init__(self):
        """Initialize the instance."""
        self.n_support_ = np.array([1, 1])
        self.support_vectors_ = np.array([[1.0]])
        self.dual_coef_ = np.array([[0.5, -0.5]])
        self.intercept_ = np.array([0.1])
        self.probA_ = np.array([0.1])
        self.probB_ = np.array([0.2])
        self.classes_ = np.array(["A", "B"])
        self.kernel = "rbf"
        self._gamma = 0.5
        self.coef0 = 0.0
        self.degree = 3


def test_svm_classifier_strings():
    """Tests the svm classifier strings functionality."""
    g = Graph("g")
    _convert_svm_classifier(MockSVCStrings(), ["input"], g)
    assert "classlabels_strings" in g.nodes[-2].attrs
