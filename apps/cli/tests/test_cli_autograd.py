import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import autograd_cmd


def test_autograd_cmd_default_output():
    args = argparse.Namespace(model="dummy.onnx", output=None)
    with (
        patch("builtins.print") as mock_print,
        patch("onnx9000_cli.main.load_onnx", return_value="graph") as mock_load,
        patch("onnx9000.toolkit.training.autograd.compiler.AutogradEngine") as mock_engine,
        patch("onnx9000_cli.main.save_onnx") as mock_save,
    ):
        mock_instance = mock_engine.return_value
        mock_instance.build_backward_graph.return_value = "bw_graph"

        autograd_cmd(args)

        mock_load.assert_called_once_with("dummy.onnx")
        mock_instance.build_backward_graph.assert_called_once_with("graph")
        mock_save.assert_called_once_with("bw_graph", "dummy_bw.onnx")
        mock_print.assert_any_call("Autograd complete.")


def test_autograd_cmd_custom_output():
    args = argparse.Namespace(model="dummy.onnx", output="custom.onnx")
    with (
        patch("builtins.print"),
        patch("onnx9000_cli.main.load_onnx", return_value="graph"),
        patch("onnx9000.toolkit.training.autograd.compiler.AutogradEngine") as mock_engine,
        patch("onnx9000_cli.main.save_onnx") as mock_save,
    ):
        mock_instance = mock_engine.return_value
        mock_instance.build_backward_graph.return_value = "bw_graph"

        autograd_cmd(args)

        mock_save.assert_called_once_with("bw_graph", "custom.onnx")
