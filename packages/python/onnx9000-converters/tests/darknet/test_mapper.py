"""Tests for Darknet mapper."""

import numpy as np
from onnx9000.converters.darknet.mapper import DarknetMapper


def test_mapper_basic_conv():
    """Test mapping a basic convolutional layer."""
    layers = [
        {"type": "net", "width": "416", "height": "416", "channels": "3"},
        {
            "type": "convolutional",
            "filters": "32",
            "size": "3",
            "stride": "1",
            "pad": "1",
            "batch_normalize": "0",
            "activation": "linear",
        },
    ]
    # filters=32, channels=3, size=3 -> weights: 32 + 32*3*3*3 = 32 + 864 = 896
    weights = np.zeros(896, dtype=np.float32)
    mapper = DarknetMapper(layers, weights)
    graph = mapper.map()

    assert len(graph.nodes) == 1
    assert graph.nodes[0].op_type == "Conv"
    assert graph.nodes[0].attributes["kernel_shape"] == [3, 3]


def test_mapper_conv_bn_leaky():
    """Test mapping a convolutional layer with batch norm and leaky relu."""
    layers = [
        {"type": "net"},
        {
            "type": "convolutional",
            "filters": "16",
            "size": "1",
            "stride": "1",
            "pad": "0",
            "batch_normalize": "1",
            "activation": "leaky",
        },
    ]
    # bn requires 16 * 4 = 64
    # conv requires 16 * 3 * 1 * 1 = 48
    weights = np.zeros(64 + 48, dtype=np.float32)
    mapper = DarknetMapper(layers, weights)
    graph = mapper.map()

    assert len(graph.nodes) == 3
    assert graph.nodes[0].op_type == "Conv"
    assert graph.nodes[1].op_type == "BatchNormalization"
    assert graph.nodes[2].op_type == "LeakyRelu"


def test_mapper_route():
    """Test mapping a route layer."""
    layers = [
        {"type": "net"},
        {"type": "convolutional", "filters": "16", "size": "1"},  # 0
        {"type": "convolutional", "filters": "16", "size": "1"},  # 1
        {"type": "route", "layers": "-1, -2"},  # 2
    ]
    weights = np.zeros(1000, dtype=np.float32)
    mapper = DarknetMapper(layers, weights)
    graph = mapper.map()

    # We expect 2 Conv nodes and 1 Concat node
    concat_nodes = [n for n in graph.nodes if n.op_type == "Concat"]
    assert len(concat_nodes) == 1


def test_mapper_shortcut():
    """Test mapping a shortcut layer."""
    layers = [
        {"type": "net"},
        {"type": "convolutional", "filters": "16", "size": "1"},  # 0
        {"type": "convolutional", "filters": "16", "size": "1"},  # 1
        {"type": "shortcut", "from": "-2"},  # 2
    ]
    weights = np.zeros(1000, dtype=np.float32)
    mapper = DarknetMapper(layers, weights)
    graph = mapper.map()

    add_nodes = [n for n in graph.nodes if n.op_type == "Add"]
    assert len(add_nodes) == 1


def test_mapper_maxpool_mish():
    """Test mapping maxpool and mish."""
    layers = [
        {"type": "net"},
        {"type": "maxpool", "size": "2", "stride": "2"},
        {"type": "convolutional", "filters": "16", "size": "1", "activation": "mish"},
    ]
    weights = np.zeros(1000, dtype=np.float32)
    mapper = DarknetMapper(layers, weights)
    graph = mapper.map()

    pool_nodes = [n for n in graph.nodes if n.op_type == "MaxPool"]
    assert len(pool_nodes) == 1
    mish_nodes = [n for n in graph.nodes if n.op_type == "Mish"]
    assert len(mish_nodes) == 1


def test_mapper_yolo_and_empty():
    """Test mapping yolo layer and empty layers."""
    layers = [{"type": "net"}, {"type": "yolo"}]
    weights = np.zeros(1000, dtype=np.float32)
    mapper = DarknetMapper(layers, weights)
    graph = mapper.map()
    # YOLO currently mapped to nothing (output pass-through)
    assert len(graph.nodes) == 0


def test_mapper_empty():
    """Test mapping empty layers."""
    mapper = DarknetMapper([], np.zeros(0))
    graph = mapper.map()
    assert len(graph.nodes) == 0
