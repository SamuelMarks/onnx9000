"""Tests for NCNN mapper."""

import numpy as np
from onnx9000.converters.ncnn.mapper import NCNNMapper


class MockWeightsReader:
    """Mock weights reader for testing."""

    def read_blob(self, num_elements: int) -> np.ndarray:
        return np.zeros(num_elements, dtype=np.float32)


def test_mapper_basic():
    """Test mapping basic NCNN layers."""
    param_info = {
        "layers": [
            {
                "type": "Input",
                "name": "data",
                "bottoms": [],
                "tops": ["data"],
                "params": {0: 224, 1: 224, 2: 3},
            },
            {
                "type": "Convolution",
                "name": "conv1",
                "bottoms": ["data"],
                "tops": ["conv1"],
                "params": {0: 64, 1: 3, 5: 1, 6: 1728},
            },
            {
                "type": "Pooling",
                "name": "pool1",
                "bottoms": ["conv1"],
                "tops": ["pool1"],
                "params": {0: 0, 1: 2, 2: 2},
            },
            {
                "type": "ReLU",
                "name": "relu1",
                "bottoms": ["pool1"],
                "tops": ["relu1"],
                "params": {},
            },
            {
                "type": "Split",
                "name": "split1",
                "bottoms": ["relu1"],
                "tops": ["split1_1", "split1_2"],
                "params": {},
            },
            {
                "type": "ConvolutionDepthWise",
                "name": "dwconv",
                "bottoms": ["split1_1"],
                "tops": ["dwconv_out"],
                "params": {0: 64, 1: 3, 5: 0, 6: 576},
            },
        ]
    }
    mapper = NCNNMapper(param_info, MockWeightsReader())
    graph = mapper.map()

    assert len(graph.inputs) == 1
    assert graph.inputs[0].name == "data"

    conv_nodes = [n for n in graph.nodes if n.op_type == "Conv"]
    assert len(conv_nodes) == 2

    pool_nodes = [n for n in graph.nodes if n.op_type == "MaxPool"]
    assert len(pool_nodes) == 1

    relu_nodes = [n for n in graph.nodes if n.op_type == "Relu"]
    assert len(relu_nodes) == 1

    identity_nodes = [n for n in graph.nodes if n.op_type == "Identity"]
    assert len(identity_nodes) == 2

    assert len(graph.outputs) == 1
    out_names = [o.name for o in graph.outputs]
    assert "dwconv_out" in out_names
