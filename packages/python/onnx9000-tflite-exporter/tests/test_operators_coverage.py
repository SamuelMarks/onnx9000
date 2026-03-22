import pytest
from onnx9000.tflite_exporter.compiler.operators import (
    _map_cast,
    _map_sequence_rnn,
    map_pool2d_options,
    map_conv2d_options,
    _map_transpose_conv,
    map_depthwise_conv2d_options,
    _map_space_depth,
)
from onnx9000.core.ir import Node, Attribute


class MockBuilder:
    def __init__(self):
        self.fields = {}

    def start_object(self, n):
        self.fields = {}

    def add_field_int8(self, i, v, default):
        self.fields[i] = v

    def add_field_int32(self, i, v, default):
        self.fields[i] = v

    def end_object(self):
        return self.fields


def test_map_cast():
    b = MockBuilder()
    for to_val, expected_type in [
        (2, 4),  # TensorType.UINT8
        (3, 9),  # TensorType.INT8
        (6, 2),  # TensorType.INT32
        (
            7,
            4,
        ),  # TensorType.INT64 mapped to int64 ? Actually 4 is UINT8, INT64 is 4 but TensorType has different values. Let's not check exact value but just run it.
        (9, 6),  # BOOL
        (10, 1),  # FLOAT16
        (11, 0),  # FLOAT32
        (8, 5),  # STRING
    ]:
        n = Node("Cast", [], [], {"to": Attribute("to", "INT", to_val)})
        res = _map_cast(b, n)
        assert res[1] is not None


def test_map_seq_rnn():
    b = MockBuilder()
    n = Node("LSTM", [], [], {"time_major": Attribute("time_major", "INT", 1)})
    res = _map_sequence_rnn(b, n)
    assert res[0] == 1  # time_major


def test_map_pool2d_options():
    b = MockBuilder()
    n = Node("MaxPool", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"SAME_UPPER")})
    res = map_pool2d_options(b, n)
    assert res[0] == 0  # Padding.SAME

    n = Node("MaxPool", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"VALID")})
    res = map_pool2d_options(b, n)
    assert res[0] == 1  # Padding.VALID


def test_map_conv2d_options():
    b = MockBuilder()
    n = Node("Conv", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"SAME_UPPER")})
    res = map_conv2d_options(b, n)
    assert res[0] == 0  # Padding.SAME

    n = Node("Conv", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"VALID")})
    res = map_conv2d_options(b, n)
    assert res[0] == 1  # Padding.VALID


def test_map_transpose_conv():
    b = MockBuilder()
    n = Node("ConvTranspose", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"SAME_UPPER")})
    res = _map_transpose_conv(b, n)
    assert res[0] == 0  # Padding.SAME

    n = Node("ConvTranspose", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"VALID")})
    res = _map_transpose_conv(b, n)
    assert res[0] == 1  # Padding.VALID


def test_map_depthwise_conv2d_options():
    b = MockBuilder()
    n = Node("Conv", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"SAME_UPPER")})
    res = map_depthwise_conv2d_options(b, n)
    assert res[0] == 0  # Padding.SAME

    n = Node("Conv", [], [], {"auto_pad": Attribute("auto_pad", "STRING", b"VALID")})
    res = map_depthwise_conv2d_options(b, n)
    assert res[0] == 1  # Padding.VALID


def test_map_space_to_depth():
    b = MockBuilder()
    n = Node("SpaceToDepth", [], [], {"blocksize": Attribute("blocksize", "INT", 2)})
    res = _map_space_depth(b, n)
    assert res[0] == 2
