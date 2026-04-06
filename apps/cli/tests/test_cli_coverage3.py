import subprocess
from unittest.mock import MagicMock, patch

from onnx9000_cli.coverage import generate_framework_snapshots


def test_pyenv_fallback():
    with patch("tempfile.TemporaryDirectory") as mock_temp:
        mock_temp.return_value.__enter__.return_value = "tmpdir"

        def fake_run(cmd, *args, **kwargs):
            mock = MagicMock()
            if "uv" in cmd and "venv" in cmd and "--python" in cmd and "/" not in cmd[3]:
                raise subprocess.CalledProcessError(1, cmd)
            if "pyenv" in cmd and "versions" in cmd:
                mock.stdout = "3.10.1\n3.10.2\n"
            elif "pyenv" in cmd and "install" in cmd:
                pass
            elif "pyenv" in cmd and "prefix" in cmd:
                mock.stdout = "/fake/pyenv/prefix"
            return mock

        with patch("subprocess.run", side_effect=fake_run):
            with patch("glob.glob", return_value=[]):
                with patch("builtins.open"):
                    with patch("json.load", return_value={"version": "1.0", "objects": ["a"]}):
                        with patch(
                            "onnx9000_cli.coverage.get_pypi_info", return_value=("1.0", "3.10")
                        ):
                            generate_framework_snapshots("snapshots_dir")


def test_pyenv_fallback_install():
    with patch("tempfile.TemporaryDirectory") as mock_temp:
        mock_temp.return_value.__enter__.return_value = "tmpdir"

        def fake_run(cmd, *args, **kwargs):
            mock = MagicMock()
            if "uv" in cmd and "venv" in cmd and "--python" in cmd and "/" not in cmd[3]:
                raise subprocess.CalledProcessError(1, cmd)
            if "pyenv" in cmd and "versions" in cmd:
                mock.stdout = "3.9.0\n"  # Missing 3.10
            elif "pyenv" in cmd and "prefix" in cmd:
                mock.stdout = "/fake/pyenv/prefix"
            return mock

        with patch("subprocess.run", side_effect=fake_run):
            with patch("glob.glob", return_value=[]):
                with patch("builtins.open"):
                    with patch("onnx9000_cli.coverage.get_pypi_info", return_value=("1.0", "3.10")):
                        generate_framework_snapshots("snapshots_dir")


def test_pyenv_fallback_fails():
    with patch("tempfile.TemporaryDirectory") as mock_temp:
        mock_temp.return_value.__enter__.return_value = "tmpdir"

        def fake_run(cmd, *args, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        with patch("subprocess.run", side_effect=fake_run):
            with patch("glob.glob", return_value=[]):
                with patch("builtins.open"):
                    with patch("onnx9000_cli.coverage.get_pypi_info", return_value=("1.0", "3.10")):
                        generate_framework_snapshots("snapshots_dir")
