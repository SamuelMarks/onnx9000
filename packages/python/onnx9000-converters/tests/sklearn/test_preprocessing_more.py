"""Tests the preprocessing more module functionality."""

import pytest
from onnx9000.converters.sklearn.preprocessing import (
    convert_label_encoder,
    convert_normalizer,
    convert_one_hot_encoder,
    convert_robust_scaler,
)
from onnx9000.core.ir import Graph


class MockRobustScaler:
    """Represents the Mock Robust Scaler class."""

    def __init__(self):
        """Initialize the instance."""
        import numpy as np

        self.center_ = np.array([0.5])
        self.scale_ = np.array([2.0])


def test_robust_scaler():
    """Tests the robust scaler functionality."""
    g = Graph("g")
    convert_robust_scaler(MockRobustScaler(), ["input"], g)
    assert "offset" in g.nodes[-1].attrs


class MockNormalizerL2:
    """Represents the Mock Normalizer L2 class."""

    norm = "l2"


class MockNormalizerMax:
    """Represents the Mock Normalizer Max class."""

    norm = "max"


def test_normalizer():
    """Tests the normalizer functionality."""
    g = Graph("g")
    convert_normalizer(MockNormalizerL2(), ["input"], g)
    convert_normalizer(MockNormalizerMax(), ["input"], g)


class MockOHEInts:
    """Represents the Mock O H E Ints class."""

    categories_ = [[0, 1]]


class MockOHEStrings:
    """Represents the Mock O H E Strings class."""

    categories_ = [["A", "B"]]


def test_ohe():
    """Tests the ohe functionality."""
    g = Graph("g")
    convert_one_hot_encoder(MockOHEInts(), ["input"], g)
    convert_one_hot_encoder(MockOHEStrings(), ["input"], g)


class MockLabelEncoderInts:
    """Represents the Mock Label Encoder Ints class."""

    classes_ = [0, 1]


def test_label_encoder_ints():
    """Tests the label encoder ints functionality."""
    g = Graph("g")
    convert_label_encoder(MockLabelEncoderInts(), ["input"], g)
    assert "classes_int64s" in g.nodes[-1].attrs


from onnx9000.converters.sklearn.preprocessing import convert_binarizer


class MockNormalizerL1:
    """Represents the Mock Normalizer L1 class."""

    norm = "l1"


def test_normalizer_l1():
    """Tests the normalizer l1 functionality."""
    g = Graph("g")
    convert_normalizer(MockNormalizerL1(), ["input"], g)
    assert g.nodes[-1].attrs["norm"].value == "L1"


class MockBinarizer:
    """Represents the Mock Binarizer class."""

    threshold = 0.5


def test_binarizer():
    """Tests the binarizer functionality."""
    g = Graph("g")
    convert_binarizer(MockBinarizer(), ["input"], g)
    assert g.nodes[-1].attrs["threshold"].value == 0.5


def test_label_encoder_strings():
    """Tests the label encoder strings functionality."""

    class MockLabelEncoderStrings:
        """Represents the MockLabelEncoderStrings class and its associated logic."""

        classes_ = ["A", "B"]

    g = Graph("g")
    convert_label_encoder(MockLabelEncoderStrings(), ["input"], g)
    assert "classes_strings" in g.nodes[-1].attrs
