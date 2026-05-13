import argparse
from unittest.mock import patch

from onnx9000_cli.main import edit_cmd


def test_edit_cmd_not_found():
    args = argparse.Namespace(model="dummy.onnx")
    with patch("os.path.exists", return_value=False), patch("builtins.print") as mock_print:
        edit_cmd(args)
    mock_print.assert_any_call("Modifier UI not found in monorepo.")


def test_edit_cmd_exists():
    args = argparse.Namespace(model="dummy.onnx")
    with patch("os.path.exists", return_value=True), patch("subprocess.run") as mock_run:
        edit_cmd(args)
    mock_run.assert_called_once()
    assert "pnpm" in mock_run.call_args[0][0]
    assert "dev" in mock_run.call_args[0][0]


def test_edit_cmd_interrupt():
    args = argparse.Namespace(model="dummy.onnx")
    with (
        patch("os.path.exists", return_value=True),
        patch("subprocess.run", side_effect=KeyboardInterrupt),
    ):
        edit_cmd(args)
