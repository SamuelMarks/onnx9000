"""Tests for compiling a training graph using mocks."""

from unittest.mock import patch

from onnx9000.core.ir import Graph
from onnx9000.toolkit.training import compile_training_graph


@patch("onnx9000.toolkit.training.AOTBuilder")
def test_compile_training_graph_mock(
    mock_aot_builder: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test the compilation of a training graph with a mocked AOTBuilder."""
    g = Graph("mock")

    def loss_fn(
        a: int | float | str | bool | bytes | list | dict | tuple | None,
        b: int | float | str | bool | bytes | list | dict | tuple | None,
        c: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> str:
        """Mock loss function."""
        return "loss"

    loss_fn(None, None, None)

    def opt_fn(
        a: int | float | str | bool | bytes | list | dict | tuple | None,
        b: int | float | str | bool | bytes | list | dict | tuple | None,
        c: int | float | str | bool | bytes | list | dict | tuple | None,
        d: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> None:
        """Mock optimizer function."""
        return None

    opt_fn(None, None, None, None)

    mock_instance = mock_aot_builder.return_value
    mock_instance.build_training_graph.return_value = g

    res = compile_training_graph(g, loss_fn, opt_fn, "lr")
    assert res is g
    mock_instance.build_training_graph.assert_called_once_with(loss_fn, opt_fn, "lr")
