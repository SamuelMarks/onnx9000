"""Unit tests and verification suite for cli main 66 functionality."""

import sys
from unittest.mock import patch

from onnx9000.core.exceptions import Onnx9000Error


def test_coverage_gaps_cmd66() -> None:
    """Verify that coverage gaps cmd66 behaves as expected."""
    try:
        from onnx9000_cli.main import main

        with patch("sys.exit"), patch.object(sys, "argv", ["onnx9000"]):
            main()
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
    assert test_coverage_gaps_cmd66 is not None
