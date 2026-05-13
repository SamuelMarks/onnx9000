import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import prune_cmd, sparse_cmd


def test_sparse_cmd_missing_func():
    args = argparse.Namespace()
    with patch("sys.exit", side_effect=SystemExit) as mock_exit, patch("builtins.print"):
        try:
            sparse_cmd(args)
        except SystemExit:
            pass
        mock_exit.assert_called_once_with(1)


def test_sparse_cmd_with_func():
    mock_func = MagicMock()
    args = argparse.Namespace(sparse_func=mock_func)
    sparse_cmd(args)
    mock_func.assert_called_once_with(args)


def test_prune_cmd():
    args = argparse.Namespace(model="dummy.onnx", nodes="node1,node2", output=None)
    with (
        patch("builtins.print"),
        patch("onnx9000_cli.main.load_onnx") as mock_load,
        patch("onnx9000_cli.main.save_onnx") as mock_save,
    ):
        mock_graph = MagicMock()
        mock_node1 = MagicMock()
        mock_node1.name = "node1"
        mock_node2 = MagicMock()
        mock_node2.name = "node3"
        mock_graph.nodes = [mock_node1, mock_node2]
        mock_load.return_value = mock_graph

        prune_cmd(args)

        mock_load.assert_called_once_with("dummy.onnx")
        mock_save.assert_called_once()
        assert len(mock_graph.nodes) == 1
