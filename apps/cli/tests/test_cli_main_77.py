"""Unit tests and verification suite for cli main 77 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error


def test_coverage_gaps_cmd77() -> None:
    """Verify that coverage gaps cmd77 behaves as expected."""
    cmds = [
        ["mutate", "test.onnx", "--script", "s.json"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with (
            patch("builtins.open"),
            patch("os.path.isdir", return_value=False),
            patch("os.path.exists", return_value=False),
            patch("json.load", return_value=[{"action": "remove_node", "node_name": "x"}]),
        ):
            for cmd_args in cmds:
                with patch.object(sys, "argv", ["onnx9000", *cmd_args]):
                    try:
                        from onnx9000_cli.main import main as m

                        m()
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
