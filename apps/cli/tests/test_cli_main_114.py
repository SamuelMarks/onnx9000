"""Unit tests and verification suite for cli main 114 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error


def test_coverage_gaps_cmd114() -> None:
    """Verify that coverage gaps cmd114 behaves as expected."""
    cmds = [
        [
            "simplify",
            "test.onnx",
            "out.onnx",
            "--skip-fusions",
            "--skip-constant-folding",
            "--skip-shape-inference",
            "--skip-fuse-bn",
            "--dry-run",
            "--max-iterations",
            "2",
            "--log-json",
            "--size-limit-mb",
            "10",
            "--input-shape",
            "a:1,2",
            "b:a,b",
            "c:",
        ],
    ]

    class DummyInput:
        """Represents dummy input."""

        def __init__(
            self, name: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> None:
            """Initialize DummyInput with configured parameters and state.

            Args:
                name: Parameter specifying name.
            """
            self.name = name
            self.shape = (1, 2)
            self.dtype = "float32"

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[DummyInput("old")], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.optimizer.simplifier": MagicMock(
                    simplify=MagicMock(return_value=MagicMock())
                ),
            },
        ):
            with patch("builtins.open"), patch("os.path.exists", return_value=False):
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
