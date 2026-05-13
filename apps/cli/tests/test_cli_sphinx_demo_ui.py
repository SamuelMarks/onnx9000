import argparse
from unittest.mock import patch

from onnx9000_cli.main import sphinx_demo_ui_cmd


def test_sphinx_demo_ui_cmd():
    args = argparse.Namespace()
    with patch("builtins.print") as mock_print:
        sphinx_demo_ui_cmd(args)
        mock_print.assert_any_call("Launching Sphinx Demo UI local server...")
