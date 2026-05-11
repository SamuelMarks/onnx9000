"""Tests for MXNet mapper."""

import numpy as np
from onnx9000.converters.mxnet.mapper import MXNetMapper


def test_mapper_mxnet_basic():
    """Test mapping MXNet symbol dict to Graph."""
    symbol_info = {
        "nodes": [
            {"op": "null", "name": "data"},
            {"op": "null", "name": "conv1_weight"},
            {
                "op": "Convolution",
                "name": "conv1",
                "inputs": [[0, 0, 0], [1, 0, 0]],
                "attrs": {"kernel": "(3, 3)", "stride": "(1, 1)", "pad": "(1, 1)"},
            },
            {
                "op": "Pooling",
                "name": "pool1",
                "inputs": [[2, 0, 0]],
                "attrs": {
                    "pool_type": "max",
                    "kernel": "(2, 2)",
                    "stride": "(2, 2)",
                    "pad": "(0, 0)",
                },
            },
            {
                "op": "Activation",
                "name": "relu1",
                "inputs": [[3, 0, 0]],
                "attrs": {"act_type": "relu"},
            },
            {"op": "null", "name": "fc1_weight"},
            {
                "op": "FullyConnected",
                "name": "fc1",
                "inputs": [[4, 0, 0], [5, 0, 0]],
                "attrs": {"num_hidden": "1000"},
            },
            {"op": "Flatten", "name": "flatten0", "inputs": [[6, 0, 0]], "attrs": {}},
        ],
        "heads": [[7, 0, 0]],
    }

    weights = {
        "conv1_weight": np.zeros((64, 3, 3, 3), dtype=np.float32),
        "fc1_weight": np.zeros((1000, 1000), dtype=np.float32),
    }

    mapper = MXNetMapper(symbol_info, weights)
    graph = mapper.map()

    assert len(graph.inputs) == 1
    assert graph.inputs[0].name == "data"

    # Conv, MaxPool, Relu, Gemm, Flatten
    assert len(graph.nodes) == 5
    assert graph.nodes[0].op_type == "Conv"
    assert graph.nodes[1].op_type == "MaxPool"
    assert graph.nodes[2].op_type == "Relu"
    assert graph.nodes[3].op_type == "Gemm"
    assert graph.nodes[4].op_type == "Flatten"

    assert len(graph.outputs) == 1
    assert graph.outputs[0].name == "flatten0"
