import argparse
from unittest.mock import MagicMock, patch
import pytest
from onnx9000_cli.main import convert_cmd


def test_paddle_convert_with_params():
    args = argparse.Namespace(src="model.pdmodel", to="onnx", output="out.onnx")
    setattr(args, "from", "paddle")
    with patch("os.path.isdir", return_value=False):
        with patch("os.path.exists", return_value=True):  # params exist
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b"data"
                with patch(
                    "onnx9000.converters.paddle.api.convert_paddle_to_onnx",
                    return_value=MagicMock(),
                ) as mock_conv:
                    with patch("onnx9000.core.exporter.export_graph"):
                        convert_cmd(args)
                        assert mock_conv.call_count == 1


def test_paddle_convert_without_params():
    args = argparse.Namespace(src="model.pdmodel", to="onnx", output="out.onnx")
    setattr(args, "from", "paddle")
    with patch("os.path.isdir", return_value=False):
        with patch("os.path.exists", return_value=False):  # params do NOT exist
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b"data"
                with patch(
                    "onnx9000.converters.paddle.api.convert_paddle_to_onnx",
                    return_value=MagicMock(),
                ) as mock_conv:
                    with patch("onnx9000.core.exporter.export_graph"):
                        convert_cmd(args)
                        assert mock_conv.call_count == 1


def test_paddle_convert_isdir():
    args = argparse.Namespace(src="model_dir", to="onnx", output="out.onnx")
    setattr(args, "from", "paddle")
    with patch("os.path.isdir", return_value=True):
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b"data"
                with patch(
                    "onnx9000.converters.paddle.api.convert_paddle_to_onnx",
                    return_value=MagicMock(),
                ) as mock_conv:
                    with patch("onnx9000.core.exporter.export_graph"):
                        convert_cmd(args)


def test_paddle_convert_isdir_fallback():
    args = argparse.Namespace(src="model_dir", to="onnx", output="out.onnx")
    setattr(args, "from", "paddle")
    with patch("os.path.isdir", return_value=True):
        with patch("os.path.exists", side_effect=[False, False, True, True]):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b"data"
                with patch(
                    "onnx9000.converters.paddle.api.convert_paddle_to_onnx",
                    return_value=MagicMock(),
                ) as mock_conv:
                    with patch("onnx9000.core.exporter.export_graph"):
                        convert_cmd(args)
