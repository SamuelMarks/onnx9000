import argparse
import pytest
import sys
from unittest.mock import MagicMock, patch

# Mock the entire onnx9000_optimum namespace before importing functions that might rely on it
mock_optimum = MagicMock()
sys.modules["onnx9000_optimum"] = mock_optimum
sys.modules["onnx9000_optimum.export"] = mock_optimum.export
sys.modules["onnx9000_optimum.optimize"] = mock_optimum.optimize
sys.modules["onnx9000_optimum.quantize"] = mock_optimum.quantize

from onnx9000_cli.main import (
    optimum_export_cmd,
    optimum_optimize_cmd,
    optimum_quantize_cmd,
    optimum_cmd,
)


@patch("onnx9000_optimum.export.export_model")
def test_optimum_export_cmd(mock_export):
    args = argparse.Namespace(
        model_id="test_model",
        task="test_task",
        opset=14,
        device="cpu",
        cache_dir="/tmp",
        split=False,
    )
    optimum_export_cmd(args)
    mock_export.assert_called_once()


@patch("onnx9000_optimum.optimize.optimize_model")
def test_optimum_optimize_cmd(mock_optimize):
    args = argparse.Namespace(
        model="test_model", level="O2", disable_fusion=False, optimize_size=True
    )
    optimum_optimize_cmd(args)
    mock_optimize.assert_called_once()


@patch("onnx9000_optimum.quantize.quantize_model")
def test_optimum_quantize_cmd(mock_quantize):
    args = argparse.Namespace(
        model="test_model", quantize="dynamic", gptq_bits=4, gptq_group_size=128
    )
    optimum_quantize_cmd(args)
    mock_quantize.assert_called_once()


def test_optimum_cmd_missing_subcmd(capsys):
    args = argparse.Namespace()
    with pytest.raises(SystemExit):
        optimum_cmd(args)
    captured = capsys.readouterr()
    assert "Missing optimum subcommand" in captured.out


def test_optimum_cmd_has_subcmd():
    def dummy_func(a):
        a.called = True

    args = argparse.Namespace(optimum_func=dummy_func)
    optimum_cmd(args)
    assert args.called
