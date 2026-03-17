import pytest
from onnx9000.core.ir import Graph
from onnx9000.converters.sklearn.linear import _convert_linear_classifier


class MockEstimator:
    def __init__(self):
        import numpy as np

        self.coef_ = np.array([[1.0, 2.0]])
        self.intercept_ = np.array([0.5])
        self.classes_ = ["A", "B"]


def test_linear_classifier_strings():
    g = Graph("g")
    _convert_linear_classifier(MockEstimator(), ["input"], g)
    assert "classlabels_strings" in g.nodes[-2].attrs


class MockEstimatorInts:
    def __init__(self):
        import numpy as np

        self.coef_ = np.array([[1.0, 2.0]])
        self.intercept_ = np.array([0.5])
        self.classes_ = [0, 1]


def test_linear_classifier_ints():
    g = Graph("g")
    _convert_linear_classifier(MockEstimatorInts(), ["input"], g)
    assert "classlabels_int64s" in g.nodes[-2].attrs
