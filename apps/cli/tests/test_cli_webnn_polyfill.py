"""Unit tests and verification suite for cli webnn polyfill functionality."""

import argparse
from unittest.mock import patch

from onnx9000_cli.main import webnn_polyfill_cmd


def test_webnn_polyfill_cmd() -> None:
    """Verify that webnn polyfill cmd behaves as expected."""
    args = argparse.Namespace()
    with patch("builtins.print") as mock_print:
        webnn_polyfill_cmd(args)

        mock_print.assert_any_call("Testing WebNN Polyfill compatibility...")
        mock_print.assert_any_call("WebNN Polyfill environment verified.")
