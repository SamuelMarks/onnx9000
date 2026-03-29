"""Tests for packages/python/onnx9000-tflite-exporter/tests/test_cli.py."""

import pytest
from unittest.mock import patch, MagicMock
from onnx9000.tflite_exporter.cli import main


@patch("onnx9000.tflite_exporter.cli.argparse.ArgumentParser.parse_args")
@patch("builtins.print")
def test_cli_parsing(mock_print, mock_parse_args):
    """Test cli parsing."""
    mock_args = MagicMock()
    mock_args.input = "dummy.onnx"
    mock_args.output = "dummy.tflite"
    mock_args.keep_nchw = True
    mock_args.int8 = True
    mock_args.fp16 = False
    mock_args.disable_optimization = False
    mock_args.external_weights = ""
    mock_args.progress = False
    mock_args.micro = False
    mock_parse_args.return_value = mock_args
    main(["dummy.onnx", "--keep-nchw", "--int8"])
    mock_parse_args.assert_called_once_with(["dummy.onnx", "--keep-nchw", "--int8"])
    mock_print.assert_any_call("[onnx2tf] Loading ONNX model from dummy.onnx...")
    mock_print.assert_any_call("[onnx2tf] Compiling to TFLite... (keep_nchw=True, quant_mode=int8)")


@patch("onnx9000.tflite_exporter.cli.argparse.ArgumentParser.parse_args")
@patch("builtins.print")
def test_cli_fp16(mock_print, mock_parse_args):
    """Test cli fp16."""
    mock_args = MagicMock()
    mock_args.input = "dummy.onnx"
    mock_args.output = "dummy.tflite"
    mock_args.keep_nchw = False
    mock_args.int8 = False
    mock_args.fp16 = True
    mock_args.batch = 4
    mock_args.disable_optimization = False
    mock_args.external_weights = ""
    mock_args.progress = False
    mock_args.micro = False
    mock_parse_args.return_value = mock_args
    main(["dummy.onnx", "--fp16", "-b", "4"])
    mock_print.assert_any_call("[onnx2tf] Overriding dynamic batch size to 4")
    mock_print.assert_any_call(
        "[onnx2tf] Compiling to TFLite... (keep_nchw=False, quant_mode=fp16)"
    )


@patch("onnx9000.tflite_exporter.cli.argparse.ArgumentParser.parse_args")
@patch("builtins.print")
def test_cli_disable_opt(mock_print, mock_parse_args):
    """Test cli disable opt."""
    mock_args = MagicMock()
    mock_args.input = "dummy.onnx"
    mock_args.output = "dummy.tflite"
    mock_args.keep_nchw = False
    mock_args.int8 = False
    mock_args.fp16 = False
    mock_args.batch = None
    mock_args.external_weights = ""
    mock_args.progress = False
    mock_args.micro = False
    mock_args.disable_optimization = True
    mock_parse_args.return_value = mock_args
    main(["dummy.onnx", "--disable-optimization"])
    mock_print.assert_any_call("[onnx2tf] Disabling layout and math optimizations...")


@patch("onnx9000.tflite_exporter.cli.argparse.ArgumentParser.parse_args")
@patch("builtins.print")
def test_cli_misc_flags(mock_print, mock_parse_args):
    """Test cli misc flags."""
    mock_args = MagicMock()
    mock_args.input = "dummy.onnx"
    mock_args.output = "dummy.tflite"
    mock_args.keep_nchw = False
    mock_args.int8 = False
    mock_args.fp16 = False
    mock_args.batch = None
    mock_args.external_weights = "weights.bin"
    mock_args.progress = True
    mock_args.micro = True
    mock_args.disable_optimization = False
    mock_args.saved_model = True
    mock_parse_args.return_value = mock_args
    main(
        [
            "dummy.onnx",
            "--external-weights",
            "weights.bin",
            "--progress",
            "--micro",
            "--saved-model",
        ]
    )
    mock_print.assert_any_call("[onnx2tf] Using external weights from weights.bin")
    mock_print.assert_any_call("[onnx2tf] Enabling build progress tracking...")
    mock_print.assert_any_call(
        "[onnx2tf] Warning: Generating TFLite Micro compatible schema (dropping optional headers)"
    )


def test_cli_empty_args():
    """Test cli empty args."""
    with patch("sys.argv", ["onnx2tf"]):
        with pytest.raises(SystemExit):
            main()


import runpy


def test_cli_main_block():
    """Test cli main block."""
    with patch("sys.argv", ["onnx2tf", "--help"]):
        try:
            runpy.run_module("onnx9000.tflite_exporter.cli", run_name="__main__")
        except SystemExit:
            return None
