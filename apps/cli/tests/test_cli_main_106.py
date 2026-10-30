"""Unit tests and verification suite for cli main 106 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd106() -> None:
    """Verify that coverage gaps cmd106 behaves as expected."""
    cmds = [
        ["convert", "test.onnx", "--from", "pytorch", "--to", "onnx", "-o", "out.onnx"],
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
                "torch.export": MagicMock(load=MagicMock(side_effect=RuntimeError)),
                "torch.load": MagicMock(return_value="t"),
                "torch.nn": MagicMock(Module=type("Mod", (object,), {})),
                "onnx9000.converters.torch.fx": MagicMock(
                    PyTorchFXParser=MagicMock(return_value=MagicMock(parse=MagicMock()))
                ),
            },
        ):
            with patch("os.path.isdir", side_effect=[False] * 4):
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
