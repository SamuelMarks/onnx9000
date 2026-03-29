"""Tests for packages/python/onnx9000-tflite-exporter/tests/test_operators_coverage.py."""

import pytest
from onnx9000.core.ir import Attribute, Node
from onnx9000.tflite_exporter.compiler.operators import (
    _map_cast,
    _map_sequence_rnn,
    _map_space_depth,
    _map_transpose_conv,
    map_conv2d_options,
    map_depthwise_conv2d_options,
    map_pool2d_options,
)


class MockBuilder:
    """MockBuilder implementation."""

    def __init__(self):
        """Perform   init   operation."""
        self.fields = {}

    def start_object(self, n):
        """Perform start object operation."""
        self.fields = {}

    def add_field_int8(self, i, v, default):
        """Perform add field int8 operation."""
        self.fields[i] = v

    def add_field_int32(self, i, v, default):
        """Perform add field int32 operation."""
        self.fields[i] = v

    def end_object(self):
        """Perform end object operation."""
        return self.fields


def test_map_cast():
    """Test map cast."""
    b = MockBuilder()
    for to_val, expected_type in [(2, 4), (3, 9), (6, 2), (7, 4), (9, 6), (10, 1), (11, 0), (8, 5)]:
        n = Node("Cast", [], [], {"to": Attribute("to", "INT", to_val)})
        res = _map_cast(b, n)
        assert res[1] is not None


def test_map_seq_rnn():
    """Test map seq rnn."""
    b = MockBuilder()
    n = Node("LSTM", [], [], {"time_major": Attribute("time_major", "INT", 1)})
    res = _map_sequence_rnn(b, n)
    assert res[0] == 1


def test_map_pool2d_options():
    """Test map pool2d options."""
    b = MockBuilder()
    n = Node("MaxPool", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"SAME_UPPER")})
    res = map_pool2d_options(b, n)
    assert res[0] == 0
    n = Node("MaxPool", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"VALID")})
    res = map_pool2d_options(b, n)
    assert res[0] == 1


def test_map_conv2d_options():
    """Test map conv2d options."""
    b = MockBuilder()
    n = Node("Conv", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"SAME_UPPER")})
    res = map_conv2d_options(b, n)
    assert res[0] == 0
    n = Node("Conv", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"VALID")})
    res = map_conv2d_options(b, n)
    assert res[0] == 1


def test_map_transpose_conv():
    """Test map transpose conv."""
    b = MockBuilder()
    n = Node("ConvTranspose", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"SAME_UPPER")})
    res = _map_transpose_conv(b, n)
    assert res[0] == 0
    n = Node("ConvTranspose", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"VALID")})
    res = _map_transpose_conv(b, n)
    assert res[0] == 1


def test_map_depthwise_conv2d_options():
    """Test map depthwise conv2d options."""
    b = MockBuilder()
    n = Node("Conv", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"SAME_UPPER")})
    res = map_depthwise_conv2d_options(b, n)
    assert res[0] == 0
    n = Node("Conv", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"VALID")})
    res = map_depthwise_conv2d_options(b, n)
    assert res[0] == 1


def test_map_space_to_depth():
    """Test map space to depth."""
    b = MockBuilder()
    n = Node("SpaceToDepth", [], [], {"blocksize": Attribute("blocksize", "INT", 2)})
    res = _map_space_depth(b, n)
    assert res[0] == 2
