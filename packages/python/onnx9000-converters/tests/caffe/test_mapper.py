"""Tests for Caffe mapper."""

import numpy as np
from onnx9000.converters.caffe.mapper import CaffeMapper


def test_mapper_basic():
    """Test mapping basic Caffe layers."""
    net_info = {
        "name": ["TestNet"],
        "input": ["data"],
        "input_dim": [1, 3, 224, 224],
        "layer": [
            {
                "name": ["conv1"],
                "type": ["Convolution"],
                "bottom": ["data"],
                "top": ["conv1"],
                "convolution_param": [
                    {"num_output": [64], "kernel_size": [3], "stride": [1], "pad": [1]}
                ],
            },
            {
                "name": ["pool1"],
                "type": ["Pooling"],
                "bottom": ["conv1"],
                "top": ["pool1"],
                "pooling_param": [{"pool": ["MAX"], "kernel_size": [2], "stride": [2]}],
            },
            {
                "name": ["ip1"],
                "type": ["InnerProduct"],
                "bottom": ["pool1"],
                "top": ["ip1"],
                "inner_product_param": [{"num_output": [1000]}],
            },
            {"name": ["relu1"], "type": ["ReLU"], "bottom": ["ip1"], "top": ["relu1"]},
        ],
    }

    weights = {
        "conv1": [np.zeros((64, 3, 3, 3), dtype=np.float32), np.zeros(64, dtype=np.float32)],
        "ip1": [
            np.zeros((1000, 64 * 112 * 112), dtype=np.float32),
            np.zeros(1000, dtype=np.float32),
        ],
    }

    mapper = CaffeMapper(net_info, weights)
    graph = mapper.map()

    assert len(graph.inputs) == 1
    assert graph.inputs[0].name == "data"
    assert graph.inputs[0].shape == (1, 3, 224, 224)

    assert len(graph.nodes) == 4
    assert graph.nodes[0].op_type == "Conv"
    assert graph.nodes[1].op_type == "MaxPool"
    assert graph.nodes[2].op_type == "Gemm"
    assert graph.nodes[3].op_type == "Relu"

    assert len(graph.outputs) == 1
    assert graph.outputs[0].name == "relu1"
