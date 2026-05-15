import argparse
from unittest.mock import patch

from onnx9000_cli.main import webgpu_cmd


def test_webgpu_cmd():
    args = argparse.Namespace(model="dummy_model.onnx")
    with patch("builtins.print") as mock_print:
        webgpu_cmd(args)

        mock_print.assert_any_call("Initializing WebGPU execution for dummy_model.onnx")
        mock_print.assert_any_call("WebGPU engine loaded.")
