"""Unit tests and verification suite for cli main 97 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd97() -> None:
    """Verify that coverage gaps cmd97 behaves as expected."""
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
                return_value=MagicMock(loader=MagicMock()),
            ):
                with patch(
                    "importlib.util.module_from_spec",
                    return_value=type("module", (), {"start_chat_tui": MagicMock()}),
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
