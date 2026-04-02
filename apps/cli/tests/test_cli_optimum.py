"""Tests for the Optimum integration commands within the CLI.

This module verifies that the ONNX Optimum-based tools (export, optimize,
and quantize) appropriately handle arguments and execute the intended internal
module functions, handling subcommand dispatch and related errors cleanly.
"""

import argparse
from unittest.mock import patch

import pytest

# Mock the entire onnx9000_optimum namespace before importing functions that might rely on it
from onnx9000_cli.main import (
    optimum_cmd,
    optimum_export_cmd,
    optimum_optimize_cmd,
    optimum_quantize_cmd,
)


@patch("onnx9000_optimum.export.export_model")
def test_optimum_export_cmd(mock_export):
    """Test the Optimum export command execution behavior.

    Verifies that the `optimum_export_cmd` parses basic arguments related to
    the Hugging Face hub (like task, opset, and device cache) and properly delegates
    to the underlying Optimum module's export utility.
    """
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
    """Ensure that the Optimum optimization command correctly delegates optimization passes.

    This test checks whether flags such as `level`, `disable_fusion`, and `optimize_size`
    are passed cleanly, resulting in a single call to the Optimum module's
    optimization logic.
    """
    args = argparse.Namespace(
        model="test_model", level="O2", disable_fusion=False, optimize_size=True
    )
    optimum_optimize_cmd(args)
    mock_optimize.assert_called_once()


@patch("onnx9000_optimum.quantize.quantize_model")
def test_optimum_quantize_cmd(mock_quantize):
    """Verify the Optimum quantization command routes correctly to the underlying APIs.

    By mocking the Optimum quantization function, this asserts that flags like
    `quantize`, `gptq_bits`, and `gptq_group_size` are supported and effectively trigger
    the requested quantization process.
    """
    args = argparse.Namespace(
        model="test_model", quantize="dynamic", gptq_bits=4, gptq_group_size=128
    )
    optimum_quantize_cmd(args)
    mock_quantize.assert_called_once()


def test_optimum_cmd_missing_subcmd(capsys):
    """Validate error handling when the base Optimum command lacks a subcommand.

    Ensures that calling `optimum_cmd` without specifying a specific functionality
    (export, optimize, quantize) results in a `SystemExit` with an informative error
    message.
    """
    args = argparse.Namespace()
    with pytest.raises(SystemExit):
        optimum_cmd(args)
    captured = capsys.readouterr()
    assert "Missing optimum subcommand" in captured.out


def test_optimum_cmd_has_subcmd():
    """Verify that the base Optimum command delegates to subcommands smoothly.

    It creates a dummy sub-function simulating a dispatched CLI subcommand,
    confirming that the core dispatcher triggers it properly.
    """

    def dummy_func(a):
        """Tests the optimum command dispatcher functionality."""
        a.called = True

    args = argparse.Namespace(optimum_func=dummy_func)
    optimum_cmd(args)
    assert args.called
