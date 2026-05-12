import argparse
from unittest.mock import patch

from onnx9000_cli.main import iree_cmd


def test_iree_cmd_compile():
    args = argparse.Namespace(iree_command="compile", model="dummy.onnx")
    with patch("builtins.print") as mock_print:
        iree_cmd(args)
    mock_print.assert_any_call("Compiling dummy.onnx via IREE MLIR pipeline...")


def test_iree_cmd_run():
    args = argparse.Namespace(iree_command="run", module="dummy.wvm")
    with patch("builtins.print") as mock_print:
        iree_cmd(args)
    mock_print.assert_any_call("Running dummy.wvm via IREE WVM...")


def test_iree_cmd_invalid():
    args = argparse.Namespace(iree_command="unknown")
    with patch("builtins.print") as mock_print:
        iree_cmd(args)
    mock_print.assert_any_call("Invalid IREE command.")
