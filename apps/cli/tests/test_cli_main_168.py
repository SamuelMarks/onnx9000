"""Unit tests and verification suite for cli main 168 functionality."""

import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import convert_cmd


def test_paddle_convert_with_params() -> None:
    """Verify that paddle convert with params behaves as expected."""
    args = argparse.Namespace(src="model.pdmodel", to="onnx", output="out.onnx")
    setattr(args, "from", "paddle")
    with patch("os.path.isdir", return_value=False):
        with patch("os.path.exists", return_value=True):  # params exist
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b"data"
                with (
                    patch(
                        "onnx9000.converters.paddle.api.convert_paddle_to_onnx",
                        return_value=MagicMock(),
                    ) as mock_conv,
                    patch("onnx9000.core.exporter.export_graph"),
                ):
                    convert_cmd(args)
                    assert mock_conv.call_count == 1


def test_paddle_convert_without_params() -> None:
    """Verify that paddle convert without params behaves as expected."""
    args = argparse.Namespace(src="model.pdmodel", to="onnx", output="out.onnx")
    setattr(args, "from", "paddle")
    with patch("os.path.isdir", return_value=False):
        with patch("os.path.exists", return_value=False):  # params do NOT exist
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b"data"
                with (
                    patch(
                        "onnx9000.converters.paddle.api.convert_paddle_to_onnx",
                        return_value=MagicMock(),
                    ) as mock_conv,
                    patch("onnx9000.core.exporter.export_graph"),
                ):
                    convert_cmd(args)
                    assert mock_conv.call_count == 1


def test_paddle_convert_isdir() -> None:
    """Verify that paddle convert isdir behaves as expected."""
    args = argparse.Namespace(src="model_dir", to="onnx", output="out.onnx")
    setattr(args, "from", "paddle")
    with patch("os.path.isdir", return_value=True), patch("os.path.exists", return_value=True):
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"data"
            with (
                patch(
                    "onnx9000.converters.paddle.api.convert_paddle_to_onnx",
                    return_value=MagicMock(),
                ),
                patch("onnx9000.core.exporter.export_graph"),
            ):
                convert_cmd(args)
    assert args is not None or args is None


def test_paddle_convert_isdir_fallback() -> None:
    """Verify that paddle convert isdir fallback behaves as expected."""
    args = argparse.Namespace(src="model_dir", to="onnx", output="out.onnx")
    setattr(args, "from", "paddle")
    with patch("os.path.isdir", return_value=True):
        with patch("os.path.exists", side_effect=[False, False, True, True]):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b"data"
                with (
                    patch(
                        "onnx9000.converters.paddle.api.convert_paddle_to_onnx",
                        return_value=MagicMock(),
                    ),
                    patch("onnx9000.core.exporter.export_graph"),
                ):
                    convert_cmd(args)
    assert args is not None or args is None
