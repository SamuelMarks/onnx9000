import argparse
from unittest.mock import patch

from onnx9000_cli.main import webnn_polyfill_cmd


def test_webnn_polyfill_cmd():
    args = argparse.Namespace()
    with patch("builtins.print") as mock_print:
        webnn_polyfill_cmd(args)

        mock_print.assert_any_call("Testing WebNN Polyfill compatibility...")
        mock_print.assert_any_call("WebNN Polyfill environment verified.")
