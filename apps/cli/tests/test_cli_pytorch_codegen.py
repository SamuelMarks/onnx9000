import argparse
from unittest.mock import MagicMock, mock_open, patch

from onnx9000_cli.main import pytorch_codegen_cmd


def test_pytorch_codegen_cmd_stdout():
    args = argparse.Namespace(model="dummy.onnx", output=None)
    mock_graph = MagicMock()

    mock_visitor = MagicMock()
    mock_visitor.generate.return_value = "class MyModel(nn.Module): pass"

    with (
        patch("onnx9000_cli.main.load_onnx", return_value=mock_graph),
        patch("onnx9000.core.parser.core.load", return_value=mock_graph),
        patch("onnx9000.core.codegen.pytorch.ONNXToPyTorchVisitor", return_value=mock_visitor),
        patch("builtins.print") as mock_print,
    ):
        pytorch_codegen_cmd(args)

    mock_print.assert_any_call("class MyModel(nn.Module): pass")


def test_pytorch_codegen_cmd_file():
    args = argparse.Namespace(model="dummy.onnx", output="out.py")
    mock_graph = MagicMock()

    mock_visitor = MagicMock()
    mock_visitor.generate.return_value = "class MyModel(nn.Module): pass"

    m_open = mock_open()
    with (
        patch("onnx9000_cli.main.load_onnx", return_value=mock_graph),
        patch("onnx9000.core.parser.core.load", return_value=mock_graph),
        patch("onnx9000.core.codegen.pytorch.ONNXToPyTorchVisitor", return_value=mock_visitor),
        patch("builtins.open", m_open),
        patch("builtins.print"),
    ):
        pytorch_codegen_cmd(args)

    m_open.assert_called_once_with("out.py", "w")
    m_open().write.assert_called_once_with("class MyModel(nn.Module): pass")
