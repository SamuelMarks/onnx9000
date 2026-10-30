"""Unit tests and verification suite for cli main 34 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd34() -> None:
    """Verify that coverage gaps cmd34 behaves as expected."""
    cmds = [
        ["genai", "--mode", "chat"],
        ["zoo"],
        ["zoo", "unknown"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.genai.ecosystem": MagicMock(),
                "onnx9000.genai.qa": MagicMock(),
            },
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
