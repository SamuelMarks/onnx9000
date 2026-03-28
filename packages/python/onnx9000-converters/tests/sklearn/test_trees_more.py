"""Tests the trees more module functionality."""

import numpy as np
import pytest
from onnx9000.converters.sklearn.trees import _convert_tree_classifier
from onnx9000.core.ir import Graph


class MockTree:
    """Represents the Mock Tree class."""

    def __init__(self):
        """Initialize the instance."""
        self.classes_ = np.array(["A", "B"])


def test_tree_classifier_strings():
    """Tests the tree classifier strings functionality."""
    g = Graph("g")
    _convert_tree_classifier(MockTree(), ["input"], g)
    assert "classlabels_strings" in g.nodes[-2].attrs
