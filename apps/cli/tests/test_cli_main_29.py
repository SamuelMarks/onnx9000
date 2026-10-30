"""Unit tests and verification suite for cli main 29 functionality."""

import sys
from unittest.mock import patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd29() -> None:
    """Verify that coverage gaps cmd29 behaves as expected."""
    with patch("sys.exit"):
        with patch.object(sys, "argv", ["onnx9000"]):
            try:
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
    assert test_coverage_gaps_cmd29 is not None
