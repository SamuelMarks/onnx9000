"""Tests the integration module functionality."""

import time

import pytest
from onnx9000.converters.paddle.api import convert_paddle_to_onnx
from onnx9000.converters.paddle.parsers import PaddleBlock, PaddleGraph, PaddleNode


def test_integration_paddle_linear_regression() -> None:
    """Tests the integration paddle linear regression functionality."""
    from onnx9000.converters.paddle.api import _convert_paddle_graph
    from onnx9000.converters.paddle.parsers import PaddleBlock, PaddleGraph, PaddleNode

    ops = [
        PaddleNode("feed", "feed", outputs={"Out": ["in1"]}),
        PaddleNode(
            "fc", "fc", inputs={"X": ["in1"], "Y": ["w1"], "Bias": ["b1"]}, outputs={"Out": ["fc1"]}
        ),
        PaddleNode(
            "sub",
            "elementwise_sub",
            inputs={"X": ["fc1"], "Y": ["target"]},
            outputs={"Out": ["sub1"]},
        ),
        PaddleNode("sqr", "square", inputs={"X": ["sub1"]}, outputs={"Out": ["sqr1"]}),
        PaddleNode("rm", "reduce_mean", inputs={"X": ["sqr1"]}, outputs={"Out": ["loss"]}),
        PaddleNode("fetch", "fetch", inputs={"X": ["loss"]}),
    ]
    b = PaddleBlock(0, -1, {}, ops)
    pg = PaddleGraph([b])
    graph = _convert_paddle_graph(pg)
    op_types = [n.op_type for n in graph.nodes]
    assert "MatMul" in op_types
    assert "Add" in op_types
    assert "Sub" in op_types
    assert "Mul" in op_types
    assert "ReduceMean" in op_types


def test_integration_paddle_resnet_mock() -> None:
    """Tests the integration paddle resnet mock functionality."""
    from onnx9000.converters.paddle.api import _convert_paddle_graph

    start_time = time.time()
    ops = [PaddleNode("n", "feed", outputs={"Out": ["in1"]})]
    for i in range(50):
        prev = f"conv_{i - 1}" if i > 0 else "in1"
        ops.append(
            PaddleNode(
                f"c{i}",
                "conv2d",
                inputs={"Input": [prev], "Filter": [f"w{i}"]},
                outputs={"Out": [f"conv_{i}"]},
            )
        )
        ops.append(
            PaddleNode(
                f"bn{i}", "batch_norm", inputs={"X": [f"conv_{i}"]}, outputs={"Y": [f"bn_{i}"]}
            )
        )
        ops.append(
            PaddleNode(f"r{i}", "relu", inputs={"X": [f"bn_{i}"]}, outputs={"Out": [f"relu_{i}"]})
        )
    b = PaddleBlock(0, -1, {}, ops)
    pg = PaddleGraph([b])
    graph = _convert_paddle_graph(pg)
    end_time = time.time()
    assert len(graph.nodes) >= 150
    assert end_time - start_time < 1.0


def test_integration_paddle_ocr_mock() -> None:
    """Tests the integration paddle ocr mock functionality."""
    from onnx9000.converters.paddle.api import _convert_paddle_graph

    ops = [
        PaddleNode("n", "feed", outputs={"Out": ["in"]}),
        PaddleNode("c", "conv2d", inputs={"Input": ["in"]}, outputs={"Out": ["conv1"]}),
        PaddleNode("r", "relu", inputs={"X": ["conv1"]}, outputs={"Out": ["relu1"]}),
        PaddleNode("lstm", "lstm", inputs={"Input": ["relu1"]}, outputs={"Out": ["lstm1"]}),
        PaddleNode("db", "sigmoid", inputs={"X": ["lstm1"]}, outputs={"Out": ["prob_map"]}),
    ]
    pg = PaddleGraph([PaddleBlock(0, -1, {}, ops)])
    graph = _convert_paddle_graph(pg)
    op_types = [n.op_type for n in graph.nodes]
    assert "Conv" in op_types
    assert "Relu" in op_types
    assert "LSTM" in op_types
    assert "Sigmoid" in op_types


def test_paddle_dynamic_shapes_and_lod() -> None:
    """Tests the paddle dynamic shapes and lod functionality."""
    from onnx9000.converters.paddle.api import _convert_paddle_graph

    ops = [
        PaddleNode("n", "feed", outputs={"Out": ["in"]}),
        PaddleNode("lod", "lod_array_length", inputs={"X": ["in"]}, outputs={"Out": ["len"]}),
        PaddleNode("rnn", "rnn", inputs={"Input": ["in"]}, outputs={"Out": ["seq_out"]}),
    ]
    pg = PaddleGraph([PaddleBlock(0, -1, {}, ops)])
    graph = _convert_paddle_graph(pg)
    op_types = [n.op_type for n in graph.nodes]
    assert "SequenceLength" in op_types
    assert "RNN" in op_types


def test_paddle_security_audit_parsing() -> None:
    """Tests the paddle security audit parsing functionality."""
    with pytest.raises(Exception):
        convert_paddle_to_onnx(b"\xff\xff\xff\xff")
