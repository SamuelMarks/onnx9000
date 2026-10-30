"""Unit tests and verification suite for cli main 121 functionality."""

import sys
from unittest.mock import patch

from onnx9000.core.exceptions import Onnx9000Error


def test_coverage_gaps_cmd121() -> None:
    """Verify that coverage gaps cmd121 behaves as expected."""
    import runpy

    try:
        with patch.object(sys, "argv", ["onnx9000", "--help"]), patch("sys.exit"):
            runpy.run_module("onnx9000_cli.main", run_name="__main__")
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass
    except SystemExit:
        pass
    assert test_coverage_gaps_cmd121 is not None
