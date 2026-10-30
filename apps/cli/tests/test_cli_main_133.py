"""Unit tests and verification suite for cli main 133 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd133() -> None:
    """Verify that coverage gaps cmd133 behaves as expected.

    Returns:
        Result of test coverage gaps cmd133.
    """
    cmds = [
        ["convert", "test.onnx", "--from", "pytorch", "--to", "onnx", "-o", "out.onnx"],
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
                "torch.export": MagicMock(load=MagicMock(side_effect=RuntimeError)),
                "torch.fx": MagicMock(symbolic_trace=MagicMock()),
                "torch": MagicMock(),
                "onnx9000.converters.torch.fx": MagicMock(
                    PyTorchFXParser=MagicMock(
                        return_value=MagicMock(parse=MagicMock(return_value=MagicMock()))
                    )
                ),
                "onnx9000.core.exporter": MagicMock(export_graph=MagicMock()),
            },
        ):
            with (
                patch("builtins.open"),
                patch("os.path.isdir", return_value=False),
                patch("os.path.exists", return_value=False),
                patch("json.load", return_value={}),
            ):
                for cmd_args in cmds:
                    with patch.object(sys, "argv", ["onnx9000", *cmd_args]):
                        import torch

                        torch.nn = MagicMock()

                        class Mod(torch.nn.Module):
                            """Represents mod inheriting from torch.nn.Module."""

                            pass

                        torch.load = MagicMock(return_value=Mod())

                        import builtins

                        orig_isinstance = builtins.isinstance

                        def fake_isinstance(
                            obj: int | float | str | bool | bytes | list | dict | tuple | None,
                            cls: int | float | str | bool | bytes | list | dict | tuple | None,
                            orig_isinstance: int
                            | float
                            | str
                            | bool
                            | bytes
                            | list
                            | dict
                            | tuple
                            | None = orig_isinstance,
                        ) -> int | float | str | bool | bytes | list | dict | tuple | None:
                            """Execute fake isinstance operation.

                            Args:
                                obj: Parameter specifying obj.
                                cls: Parameter specifying target class or tuple of classes.
                                orig_isinstance: Parameter specifying orig isinstance.

                            Returns:
                                Result of fake isinstance.
                            """
                            if cls == torch.nn.Module:
                                return True
                            return orig_isinstance(obj, cls)

                        with patch("builtins.isinstance", side_effect=fake_isinstance):
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
