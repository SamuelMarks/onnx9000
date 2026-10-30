"""Unit tests and verification suite for cli sphinx demo ui functionality."""

import argparse
from unittest.mock import patch

from onnx9000_cli.main import sphinx_demo_ui_cmd


def test_sphinx_demo_ui_cmd() -> None:
    """Verify that sphinx demo ui cmd behaves as expected."""
    args = argparse.Namespace()
    with patch("builtins.print") as mock_print:
        sphinx_demo_ui_cmd(args)
        mock_print.assert_any_call("Launching Sphinx Demo UI local server...")
