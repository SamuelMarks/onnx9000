import argparse
import os
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main, transformers_cmd


def test_transformers_cmd_success(tmp_path):
    base_dir = tmp_path / "apps" / "cli"
    dist_dir = base_dir / "dist"
    dist_dir.mkdir(parents=True)
    index_js = dist_dir / "index.js"
    index_js.touch()

    # Mock __file__ inside transformers_cmd
    mock_file = str(base_dir / "src" / "onnx9000_cli" / "main.py")

    with patch("onnx9000_cli.main.__file__", mock_file):
        with patch("subprocess.run") as mock_run:
            args = argparse.Namespace(task="text-classification", inputs=["hello", "world"])
            transformers_cmd(args)
            mock_run.assert_called_once_with(
                ["node", str(index_js), "transformers", "text-classification", "hello", "world"],
                check=True,
            )


def test_transformers_cmd_no_inputs(tmp_path):
    base_dir = tmp_path / "apps" / "cli"
    dist_dir = base_dir / "dist"
    dist_dir.mkdir(parents=True)
    index_js = dist_dir / "index.js"
    index_js.touch()

    mock_file = str(base_dir / "src" / "onnx9000_cli" / "main.py")

    with patch("onnx9000_cli.main.__file__", mock_file):
        with patch("subprocess.run") as mock_run:
            args = argparse.Namespace(task="text-generation", inputs=[])
            transformers_cmd(args)
            mock_run.assert_called_once_with(
                ["node", str(index_js), "transformers", "text-generation"],
                check=True,
            )


def test_transformers_cmd_missing_js(tmp_path):
    base_dir = tmp_path / "apps" / "cli"
    mock_file = str(base_dir / "src" / "onnx9000_cli" / "main.py")

    with patch("onnx9000_cli.main.__file__", mock_file):
        with patch("sys.exit") as mock_exit:
            args = argparse.Namespace(task="text-generation", inputs=[])
            transformers_cmd(args)
            mock_exit.assert_any_call(1)


def test_transformers_cmd_subprocess_error(tmp_path):
    import subprocess

    base_dir = tmp_path / "apps" / "cli"
    dist_dir = base_dir / "dist"
    dist_dir.mkdir(parents=True)
    index_js = dist_dir / "index.js"
    index_js.touch()

    mock_file = str(base_dir / "src" / "onnx9000_cli" / "main.py")

    with patch("onnx9000_cli.main.__file__", mock_file):
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(2, "cmd")):
            with patch("sys.exit") as mock_exit:
                args = argparse.Namespace(task="text-generation", inputs=[])
                transformers_cmd(args)
                mock_exit.assert_called_once_with(2)


def test_main_cli_routing_transformers():
    with patch("sys.argv", ["onnx9000", "transformers", "text-classification", "hello"]):
        with patch("onnx9000_cli.main.transformers_cmd") as mock_cmd:
            try:
                main()
            except SystemExit:
                pass
            mock_cmd.assert_called_once()
