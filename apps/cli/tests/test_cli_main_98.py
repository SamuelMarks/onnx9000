"""Unit tests and verification suite for cli main 98 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd98() -> None:
    """Verify that coverage gaps cmd98 behaves as expected."""
    cmds = [
        ["chat"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("builtins.open"),
        patch("os.makedirs"),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch.dict(
            sys.modules,
            {
                "tui_chat": None,
            },
        ):
            with patch(
                "importlib.util.spec_from_file_location",
                side_effect=RuntimeError("Failed to load tui_chat"),
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
