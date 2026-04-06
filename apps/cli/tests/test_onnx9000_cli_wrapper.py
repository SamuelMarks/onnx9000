"""Tests for the ONNX9000 CLI wrapper."""

import os
import sys
from unittest import mock

import pytest

import onnx9000.__main__
import onnx9000.cli


def test_cli_wrapper_success(monkeypatch):
    """Test successful import and execution."""
    called = False

    def mock_main(*args, **kwargs):
        """Mock main."""
        nonlocal called
        called = True

    class MockCliMain:
        """Mock cli main."""

        main = mock_main

    monkeypatch.setitem(sys.modules, "onnx9000_cli.main", MockCliMain())
    monkeypatch.setitem(sys.modules, "onnx9000_cli", mock.MagicMock())

    onnx9000.cli.main()
    assert called


def test_cli_wrapper_fallback_success(monkeypatch):
    """Test fallback logic when first import fails but second succeeds."""
    if "onnx9000_cli.main" in sys.modules:
        monkeypatch.delitem(sys.modules, "onnx9000_cli.main", raising=False)

    called = False

    def mock_main(*args, **kwargs):
        """Mock main."""
        nonlocal called
        called = True

    original_import = __import__

    import_calls = 0

    def side_effect_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Side effect import."""
        nonlocal import_calls
        if name == "onnx9000_cli.main" and fromlist:
            import_calls += 1
            if import_calls == 1:
                raise ImportError("First import fails")
            else:
                m = mock.MagicMock()
                m.main = mock_main
                return m
        return original_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=side_effect_import):
        onnx9000.cli.main()

    assert called
    assert import_calls == 2


def test_cli_wrapper_fallback_failure(monkeypatch):
    """Test fallback logic when both imports fail."""
    if "onnx9000_cli.main" in sys.modules:
        monkeypatch.delitem(sys.modules, "onnx9000_cli.main", raising=False)

    original_import = __import__

    def side_effect_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Side effect import."""
        if name == "onnx9000_cli.main" and fromlist:
            raise ImportError("Always fails")
        return original_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=side_effect_import):
        with mock.patch("sys.exit") as mock_exit:
            with mock.patch("builtins.print"):
                onnx9000.cli.main()
                mock_exit.assert_called_once_with(1)


def test_main_block():
    """Test the __main__ block logic in cli.py."""
    with mock.patch("sys.argv", ["onnx9000"]):
        with mock.patch("sys.exit"):
            with mock.patch("onnx9000_cli.main.main") as mock_real_main:
                with open(onnx9000.cli.__file__) as f:
                    code = compile(f.read(), onnx9000.cli.__file__, "exec")
                    exec(code, {"__name__": "__main__"})
                mock_real_main.assert_called_once()


def test_dunder_main_block():
    """Test the __main__ block logic in __main__.py."""
    with mock.patch("onnx9000.cli.main") as mock_main:
        with mock.patch("sys.argv", ["onnx9000"]):
            with open(onnx9000.__main__.__file__) as f:
                code = compile(f.read(), onnx9000.__main__.__file__, "exec")
                exec(code, {"__name__": "__main__", "__file__": onnx9000.__main__.__file__})
            mock_main.assert_called_once()
