"""Module docstring."""

from unittest.mock import MagicMock

import pytest
from onnx9000.converters.tf.keras_layers import (
    _map_keras_bidirectional,
    _map_keras_embedding,
    _map_keras_layers__attention,
    _map_keras_layers__masking,
    _map_keras_spatial_dropout,
)


class TFNodeMock:
    """Docstring for D101."""

    def __init__(self, name, op, inputs, attr):
        """Docstring for D107."""
        self.name = name
        self.op = op
        self.inputs = inputs
        self.attr = attr


def test_missing_lines():
    """Docstring for D103."""
    builder = MagicMock()
    builder.make_node.return_value = ["dummy", "dummy2"]

    # line 250: embedding
    node1 = TFNodeMock("n1", "Emb", ["i"], {})
    _map_keras_embedding(builder, node1)

    # line 273: spatial dropout
    node2 = TFNodeMock("n2", "SD", ["i"], {})
    _map_keras_spatial_dropout("1D")(builder, node2)

    # line 4767: layers__attention
    node3 = TFNodeMock("n3", "Att", ["q", "k", "v"], {})
    _map_keras_layers__attention(builder, node3)

    # line 5910: SimpleRNN
    node4 = TFNodeMock("n4", "Bi", ["i"], {"layer": {"class_name": "SimpleRNN"}})
    _map_keras_bidirectional(builder, node4)

    # line 74473: layers__masking
    node5 = TFNodeMock("n5", "Mask", ["i"], {"mask_value": 1.0})
    _map_keras_layers__masking(builder, node5)
