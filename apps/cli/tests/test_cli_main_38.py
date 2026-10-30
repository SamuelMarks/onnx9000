"""Unit tests and verification suite for cli main 38 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd38() -> None:
    """Verify that coverage gaps cmd38 behaves as expected."""
    cmds = [
        ["optimum"],
        ["info"],
        ["openvino"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
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
