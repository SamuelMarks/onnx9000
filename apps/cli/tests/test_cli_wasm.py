import argparse
from unittest.mock import patch

from onnx9000_cli.main import wasm_cmd


def test_wasm_cmd():
    args = argparse.Namespace(model="dummy_model.onnx")
    with patch("builtins.print") as mock_print:
        wasm_cmd(args)

        mock_print.assert_any_call("Initializing WebAssembly execution for dummy_model.onnx")
        mock_print.assert_any_call("WASM engine loaded.")
