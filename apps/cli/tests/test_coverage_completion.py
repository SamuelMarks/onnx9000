"""Unit tests for onnx9000 CLI entrypoint and fallback resolution."""

import runpy
import sys
from unittest.mock import patch

import pytest

import onnx9000.__main__
import onnx9000.cli


def test_main_call() -> None:
    """Tests onnx9000.__main__ module attributes."""
    with patch("onnx9000.cli.main") as mock_main:
        with patch("onnx9000.__main__.__name__", "__main__"):
            assert onnx9000.__main__.__name__ == "__main__"
            assert not mock_main.called


def test_cli_main_success() -> None:
    """Tests cli main success delegation to onnx9000_cli.main.main."""
    with patch("onnx9000_cli.main.main") as mock_cli_main:
        onnx9000.cli.main()
        mock_cli_main.assert_called_once()


def test_cli_main_fallback() -> None:
    """Tests cli main fallback when onnx9000_cli.main is unavailable."""
    with patch.dict("sys.modules", {"onnx9000_cli.main": None}):
        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["pkg1"]),
        ):
            with pytest.raises(SystemExit):
                onnx9000.cli.main()


def test_main_exec() -> None:
    """Tests running onnx9000.__main__ via runpy."""
    with patch("onnx9000.cli.main") as mock_main:
        runpy.run_module("onnx9000.__main__", run_name="__main__")
        mock_main.assert_called_once()


def test_cli_import_error() -> None:
    """Tests cli import error when workspace packages directory does not exist."""
    with patch.dict("sys.modules", {"onnx9000_cli.main": None}):
        with patch("os.path.exists", return_value=False):
            with patch("sys.exit") as mock_exit:
                onnx9000.cli.main()
                mock_exit.assert_called_with(1)


def test_cli_run_main() -> None:
    """Tests running onnx9000.cli via runpy."""
    with patch("onnx9000_cli.main.main") as mock_cli_main:
        runpy.run_module("onnx9000.cli", run_name="__main__")
        mock_cli_main.assert_called_once()


def test_cli_import_error_pkg_exists_append_sys_path() -> None:
    """Tests cli import error when packages exist and are appended to sys.path."""
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


def test_cli_fallback_success() -> None:
    """Tests cli fallback succeeding on secondary import after path discovery."""
    called = []

    with patch.dict("sys.modules", {"onnx9000_cli.main": None}):
        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["pkg1"]),
            patch("sys.path", []),
        ):

            class MockMain:
                """Mock main."""

                @staticmethod
                def main() -> None:
                    """Main."""
                    called.append(True)

            import builtins

            original_import = builtins.__import__

            def mock_import(name: str, *args: object, **kwargs: object) -> object:
                """Mock import."""
                if name == "onnx9000_cli.main":
                    if (
                        "onnx9000_cli.main" in sys.modules
                        and sys.modules["onnx9000_cli.main"] is None
                    ):
                        del sys.modules["onnx9000_cli.main"]
                        raise ImportError("first import fails")
                    sys.modules["onnx9000_cli.main"] = MockMain  # type: ignore[assignment]
                    return MockMain
                return original_import(name, *args, **kwargs)  # type: ignore[arg-type]

            with patch("builtins.__import__", side_effect=mock_import):
                onnx9000.cli.main()
                assert called == [True]


def test_cli_fallback_failure_prints_error() -> None:
    """Tests cli fallback failure prints diagnostic error and exits with code 1."""
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
