import pytest
from unittest.mock import patch
from onnx9000.core.ir import Graph
from onnx9000.toolkit.training import compile_training_graph


@patch("onnx9000.toolkit.training.AOTBuilder")
def test_compile_training_graph_mock(mock_aot_builder):
    g = Graph("mock")
    loss_fn = lambda a, b, c: "loss"
    opt_fn = lambda a, b, c, d: None

    mock_instance = mock_aot_builder.return_value
    mock_instance.build_training_graph.return_value = g

    res = compile_training_graph(g, loss_fn, opt_fn, "lr")
    assert res is g
    mock_instance.build_training_graph.assert_called_once_with(loss_fn, opt_fn, "lr")
