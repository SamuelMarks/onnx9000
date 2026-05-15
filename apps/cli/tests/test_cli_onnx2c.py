import argparse
from unittest.mock import patch

from onnx9000_cli.main import onnx2c_cmd


def test_onnx2c_cmd_default_output():
    args = argparse.Namespace(input="model.onnx", output=None)
    with patch("builtins.print") as mock_print:
        onnx2c_cmd(args)

        mock_print.assert_any_call("Converting model.onnx to C...")
        mock_print.assert_any_call("Successfully generated C code to output.c")


def test_onnx2c_cmd_custom_output():
    args = argparse.Namespace(input="model.onnx", output="custom.c")
    with patch("builtins.print") as mock_print:
        onnx2c_cmd(args)

        mock_print.assert_any_call("Converting model.onnx to C...")
        mock_print.assert_any_call("Successfully generated C code to custom.c")
