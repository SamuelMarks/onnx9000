"""Unit tests and verification suite for cli iree functionality."""

import argparse
from unittest.mock import patch

from onnx9000_cli.main import iree_cmd


def test_iree_cmd_compile() -> None:
    """Verify that iree cmd compile behaves as expected."""
    args = argparse.Namespace(iree_command="compile", model="dummy.onnx")
    with patch("builtins.print") as mock_print:
        iree_cmd(args)
    mock_print.assert_any_call("Compiling dummy.onnx via IREE MLIR pipeline...")


def test_iree_cmd_run() -> None:
    """Verify that iree cmd run behaves as expected."""
    args = argparse.Namespace(iree_command="run", module="dummy.wvm")
    with patch("builtins.print") as mock_print:
        iree_cmd(args)
    mock_print.assert_any_call("Running dummy.wvm via IREE WVM...")


def test_iree_cmd_invalid() -> None:
    """Verify that iree cmd invalid behaves as expected."""
    args = argparse.Namespace(iree_command="unknown")
    with patch("builtins.print") as mock_print:
        iree_cmd(args)
    mock_print.assert_any_call("Invalid IREE command.")
