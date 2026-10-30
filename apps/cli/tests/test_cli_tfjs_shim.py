"""Unit tests and verification suite for cli tfjs shim functionality."""

import argparse
from unittest.mock import patch

from onnx9000_cli.main import tfjs_shim_cmd


def test_tfjs_shim_cmd_stdout() -> None:
    """Verify that tfjs shim cmd stdout behaves as expected."""
    args = argparse.Namespace()
    with patch("builtins.print") as mock_print:
        tfjs_shim_cmd(args)
    mock_print.assert_any_call("TFJS Shim environment verified.")
