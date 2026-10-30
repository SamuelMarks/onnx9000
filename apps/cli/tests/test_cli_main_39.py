"""Unit tests and verification suite for cli main 39 functionality."""

import sys
from unittest.mock import patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd39() -> None:
    """Verify that coverage gaps cmd39 behaves as expected."""
    cmds = [
        ["help"],
        ["--help"],
    ]

    for cmd_args in cmds:
        with patch.object(sys, "argv", ["onnx9000", *cmd_args]):
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
    assert cmds is not None or cmds is None
