"""Module docstring."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from onnx9000 import config
from onnx9000.cli import main
from onnx9000.exceptions import (
    CompilationError,
    Onnx9000Error,
    ShapeMismatchError,
    UnsupportedOpError,
)
from onnx9000.utils.cache import clear_cache


def test_exceptions():
    """test_exceptions docstring."""
    e = UnsupportedOpError("FakeOp")
    assert e.op_type == "FakeOp"
    assert str(e) == "Operator 'FakeOp' is not supported yet."

    e2 = UnsupportedOpError("FakeOp2", "Custom msg")
    assert str(e2) == "Custom msg"

    with pytest.raises(Onnx9000Error):
        raise CompilationError()

    with pytest.raises(Onnx9000Error):
        raise ShapeMismatchError()


def test_cache(temp_dir: Path):
    """test_cache docstring."""
    config.ONNX9000_CACHE_DIR = temp_dir
    # Empty clear
    clear_cache()

    # Create something
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "test.txt").write_text("hello")
    assert temp_dir.exists()

    clear_cache()
    assert not temp_dir.exists()

    # Test error during clear
    with patch("shutil.rmtree", side_effect=Exception("Mock Error")):
        temp_dir.mkdir()
        clear_cache()  # Should just log and not raise


def test_cli(temp_dir: Path):
    """test_cli docstring."""
    test_onnx = temp_dir / "test.onnx"
    test_onnx.touch()

    with patch("onnx9000.cli.core_compile") as mock_compile:
        # Test success cpp
        with patch.object(
            sys, "argv", ["onnx9000", "compile", str(test_onnx), "--target", "cpp"]
        ):
            main()
            mock_compile.assert_called_with(str(test_onnx), target="cpp")

        # Test success wasm
        with patch.object(
            sys, "argv", ["onnx9000", "compile", str(test_onnx), "--target", "wasm"]
        ):
            main()
            mock_compile.assert_called_with(str(test_onnx), target="wasm")

        # Test error
        mock_compile.side_effect = Exception("Compile Failed")
        with patch.object(sys, "argv", ["onnx9000", "compile", str(test_onnx)]):
            with pytest.raises(SystemExit):
                main()
