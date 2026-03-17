import pytest
from onnx9000.core.ir import Graph
from onnx9000.converters.sklearn.preprocessing import (
    convert_robust_scaler,
    convert_normalizer,
    convert_one_hot_encoder,
    convert_label_encoder,
)


class MockRobustScaler:
    def __init__(self):
        import numpy as np

        self.center_ = np.array([0.5])
        self.scale_ = np.array([2.0])


def test_robust_scaler():
    g = Graph("g")
    convert_robust_scaler(MockRobustScaler(), ["input"], g)
    assert "offset" in g.nodes[-1].attrs


class MockNormalizerL2:
    norm = "l2"


class MockNormalizerMax:
    norm = "max"


def test_normalizer():
    g = Graph("g")
    convert_normalizer(MockNormalizerL2(), ["input"], g)
    convert_normalizer(MockNormalizerMax(), ["input"], g)


class MockOHEInts:
    categories_ = [[0, 1]]


class MockOHEStrings:
    categories_ = [["A", "B"]]


def test_ohe():
    g = Graph("g")
    convert_one_hot_encoder(MockOHEInts(), ["input"], g)
    convert_one_hot_encoder(MockOHEStrings(), ["input"], g)


class MockLabelEncoderInts:
    classes_ = [0, 1]


def test_label_encoder_ints():
    g = Graph("g")
    convert_label_encoder(MockLabelEncoderInts(), ["input"], g)
    assert "classes_int64s" in g.nodes[-1].attrs


from onnx9000.converters.sklearn.preprocessing import convert_binarizer


class MockNormalizerL1:
    norm = "l1"


def test_normalizer_l1():
    g = Graph("g")
    convert_normalizer(MockNormalizerL1(), ["input"], g)
    assert g.nodes[-1].attrs["norm"].value == "L1"


class MockBinarizer:
    threshold = 0.5


def test_binarizer():
    g = Graph("g")
    convert_binarizer(MockBinarizer(), ["input"], g)
    assert g.nodes[-1].attrs["threshold"].value == 0.5


def test_label_encoder_strings():
    class MockLabelEncoderStrings:
        classes_ = ["A", "B"]

    g = Graph("g")
    convert_label_encoder(MockLabelEncoderStrings(), ["input"], g)
    assert "classes_strings" in g.nodes[-1].attrs
