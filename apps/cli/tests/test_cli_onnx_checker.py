import argparse
from unittest.mock import patch

from onnx9000_cli.main import onnx_checker_cmd


def test_onnx_checker_cmd():
    args = argparse.Namespace(model="dummy.onnx")
    with patch("builtins.print") as mock_print:
        onnx_checker_cmd(args)

        mock_print.assert_any_call("Checking ONNX model dummy.onnx...")
        mock_print.assert_any_call("Model is valid.")
