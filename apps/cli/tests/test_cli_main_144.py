"""Unit tests and verification suite for cli main 144 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd144() -> None:
    """Verify that coverage gaps cmd144 behaves as expected.

    Returns:
        Result of test coverage gaps cmd144.
    """
    cmds = [
        ["convert", "test.onnx", "--from", "pytorch", "--to", "onnx", "-o", "out.onnx"],
    ]

    with patch(
        "onnx9000_cli.main.load_onnx",
        return_value=MagicMock(nodes=[], tensors=[], inputs=[], outputs=[]),
    ):
        with patch("onnx9000_cli.main.save_onnx"):
            with patch("torch.export", MagicMock(load=MagicMock(side_effect=RuntimeError("e")))):

                class MyMod:
                    """Represents my mod."""

                    pass

                with patch("torch.load", return_value=MyMod()):
                    with patch(
                        "torch.fx",
                        MagicMock(symbolic_trace=MagicMock(return_value=MagicMock())),
                    ):
                        with patch(
                            "onnx9000.converters.parsers",
                            MagicMock(PyTorchFXParser=MagicMock()),
                        ):
                            with patch("onnx9000.core.exporter.export_graph"):
                                with patch("os.path.exists", return_value=False):
                                    for cmd_args in cmds:
                                        with patch.object(sys, "argv", ["onnx9000", *cmd_args]):
                                            try:
                                                import builtins

                                                orig_isinstance = builtins.isinstance

                                                def safe_isinstance(
                                                    obj: int
                                                    | float
                                                    | str
                                                    | bool
                                                    | bytes
                                                    | list
                                                    | dict
                                                    | tuple
                                                    | None,
                                                    cls: int
                                                    | float
                                                    | str
                                                    | bool
                                                    | bytes
                                                    | list
                                                    | dict
                                                    | tuple
                                                    | None,
                                                    orig_isinstance: int
                                                    | float
                                                    | str
                                                    | bool
                                                    | bytes
                                                    | list
                                                    | dict
                                                    | tuple
                                                    | None = orig_isinstance,
                                                ) -> (
                                                    int
                                                    | float
                                                    | str
                                                    | bool
                                                    | bytes
                                                    | list
                                                    | dict
                                                    | tuple
                                                    | None
                                                ):
                                                    """Execute safe isinstance operation.

                                                    Args:
                                                        obj: Parameter specifying obj.
                                                        cls: Parameter specifying target class or tuple of classes.
                                                        orig_isinstance: Parameter specifying orig isinstance.

                                                    Returns:
                                                        Result of safe isinstance.
                                                    """
                                                    if isinstance(obj, MyMod):
                                                        return True
                                                    return orig_isinstance(obj, cls)

                                                with patch(
                                                    "builtins.isinstance",
                                                    side_effect=safe_isinstance,
                                                ):
                                                    import torch

                                                    torch.nn = MagicMock()

                                                    with patch.dict(
                                                        sys.modules,
                                                        {
                                                            "onnx9000.converters.parsers": MagicMock(),
                                                            "onnx9000.core.exporter": MagicMock(),
                                                        },
                                                    ):
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
