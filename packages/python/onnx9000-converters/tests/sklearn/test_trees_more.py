import pytest
from onnx9000.core.ir import Graph
from onnx9000.converters.sklearn.trees import _convert_tree_classifier
import numpy as np


class MockTree:
    def __init__(self):
        self.classes_ = np.array(["A", "B"])


def test_tree_classifier_strings():
    g = Graph("g")
    _convert_tree_classifier(MockTree(), ["input"], g)
    assert "classlabels_strings" in g.nodes[-2].attrs
