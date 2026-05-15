import argparse
from unittest.mock import patch

from onnx9000_cli.main import mmdnn_cmd


def test_mmdnn_cmd():
    args = argparse.Namespace(model="dummy_model")
    with patch("builtins.print") as mock_print:
        mmdnn_cmd(args)

        mock_print.assert_any_call("Converting model dummy_model via MMDNN")
        mock_print.assert_any_call("MMDNN conversion successful.")
