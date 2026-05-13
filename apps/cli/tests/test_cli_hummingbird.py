import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import hummingbird_cmd


def test_hummingbird_cmd_default_output():
    args = argparse.Namespace(model="dummy.onnx", output=None)
    with (
        patch("builtins.print"),
        patch("onnx9000_cli.main.load_onnx") as mock_load,
        patch("onnx9000.optimizer.hummingbird.engine.TranspilationEngine") as mock_engine,
        patch("onnx9000_cli.main.save_onnx") as mock_save,
        patch("onnx9000.optimizer.hummingbird.onnxml_parser.parse_onnxml_tree_ensemble"),
    ):
        mock_graph = MagicMock()
        mock_node = MagicMock()
        mock_node.op_type = "TreeEnsembleClassifier"
        mock_graph.nodes = [mock_node]
        mock_load.return_value = mock_graph

        mock_instance = mock_engine.return_value
        mock_instance.transpile.return_value = MagicMock()

        hummingbird_cmd(args)

        mock_load.assert_called_once_with("dummy.onnx")
        mock_save.assert_called_once()
