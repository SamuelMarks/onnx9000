import argparse
from unittest.mock import patch

from onnx9000_cli.main import coreml_cmd


def test_coreml_cmd_js_missing():
    args = argparse.Namespace(coreml_command="export", model="dummy.onnx")
    with (
        patch("os.path.exists", return_value=False),
        patch("sys.exit", side_effect=SystemExit) as mock_exit,
        patch("builtins.print") as mock_print,
    ):
        try:
            coreml_cmd(args)
        except SystemExit:
            pass
    mock_exit.assert_called_once_with(1)
    mock_print.assert_called_once()


def test_coreml_cmd_js_exists():
    args = argparse.Namespace(coreml_command="export", model="dummy.onnx", output="out.mlpackage")

    with patch("os.path.exists", return_value=True), patch("subprocess.run") as mock_run:
        coreml_cmd(args)

    mock_run.assert_called_once()
    assert "node" in mock_run.call_args[0][0]
