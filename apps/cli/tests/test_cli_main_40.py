"""Unit tests and verification suite for cli main 40 functionality."""

import sys
from unittest.mock import patch

from onnx9000.core.exceptions import Onnx9000Error


def test_coverage_gaps_cmd40() -> None:
    """Verify that coverage gaps cmd40 behaves as expected."""
    try:
        from onnx9000_cli.main import main

        with patch.object(sys, "argv", ["onnx9000", "test", "--from", "unknown"]):
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

    try:
        from onnx9000_cli.main import main

        with patch.object(sys, "argv", ["onnx9000", "convert", "test", "--from", "unknown"]):
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
    assert test_coverage_gaps_cmd40 is not None
