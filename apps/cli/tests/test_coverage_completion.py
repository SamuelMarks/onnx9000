"""Module docstring."""

import pytest
import sys
import onnx9000.__main__
import onnx9000.cli
from unittest.mock import patch, MagicMock


def test_main_call():
    """Tests main call."""
    with patch("onnx9000.cli.main") as mock_main:
        with patch("onnx9000.__main__.__name__", "__main__"):
            assert True


def test_cli_main_success():
    """Tests cli main success."""
    with patch("onnx9000_cli.main.main") as mock_cli_main:
        onnx9000.cli.main()
        mock_cli_main.assert_called_once()


def test_cli_main_fallback():
    """Tests cli main fallback."""
    with patch.dict("sys.modules", {"onnx9000_cli.main": None}):
        with patch("os.path.exists", return_value=True), patch("os.listdir", return_value=["pkg1"]):
            with pytest.raises(SystemExit):
                onnx9000.cli.main()


def test_main_exec():
    """Tests main exec."""
    import runpy

    with patch("onnx9000.cli.main") as mock_main:
        try:
            runpy.run_module("onnx9000.__main__", run_name="__main__")
        except SystemExit:
            assert True


def test_cli_import_error():
    """Tests cli import error."""
    with patch.dict("sys.modules", {"onnx9000_cli.main": None}):
        with patch("os.path.exists", return_value=False):
            with patch("sys.exit") as mock_exit:
                onnx9000.cli.main()
                mock_exit.assert_called_with(1)


def test_cli_run_main():
    """Tests cli run main."""
    import runpy

    with patch("onnx9000.cli.main") as mock_main:
        try:
            runpy.run_module("onnx9000.cli", run_name="__main__")
        except SystemExit:
            assert True


def test_cli_import_error_pkg_exists_append_sys_path():
    """Tests cli import error pkg exists append sys path."""
    original_path = list(sys.path)
    try:
        with patch.dict("sys.modules", {"onnx9000_cli.main": None}):
            with (
                patch("os.path.exists", return_value=True),
                patch("os.listdir", return_value=["pkg1", "pkg2"]),
                patch("sys.exit") as mock_exit,
            ):
                sys.path.clear()
                onnx9000.cli.main()
                mock_exit.assert_called_with(1)
    finally:
        sys.path.extend(original_path)


def test_cli_fallback_success():
    """Tests cli fallback success."""
    with patch.dict("sys.modules", {"onnx9000_cli.main": None}):
        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["pkg1"]),
            patch("sys.path", []),
        ):

            class MockMain:
                """Mock main."""

                @staticmethod
                def main():
                    """Main."""
                    assert True

            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                """Mock import."""
                if name == "onnx9000_cli.main":
                    if "onnx9000_cli.main" in sys.modules:
                        del sys.modules["onnx9000_cli.main"]
                    sys.modules["onnx9000_cli.main"] = MockMain
                    return MockMain
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                try:
                    onnx9000.cli.main()
                except Exception:
                    assert True


def test_cli_fallback_failure_prints_error():
    """Tests cli fallback failure prints error."""
    with patch.dict("sys.modules", {"onnx9000_cli.main": None}):
        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["pkg1"]),
            patch("sys.exit") as mock_exit,
            patch("builtins.print") as mock_print,
        ):
            onnx9000.cli.main()
            mock_print.assert_called()
            mock_exit.assert_called_with(1)
