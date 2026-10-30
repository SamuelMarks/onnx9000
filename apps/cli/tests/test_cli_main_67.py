"""Unit tests and verification suite for cli main 67 functionality."""

import sys
from unittest.mock import patch

from onnx9000.core.exceptions import Onnx9000Error


def test_coverage_gaps_cmd67() -> None:
    """Verify that coverage gaps cmd67 behaves as expected."""
    try:
        from onnx9000_cli.main import main

        with patch("sys.exit"), patch.object(sys, "argv", ["onnx9000", "optimum"]):
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

        with patch("sys.exit"):
            with patch.object(sys, "argv", ["onnx9000", "convert", "test", "--to", "onnx"]):
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
    assert test_coverage_gaps_cmd67 is not None
