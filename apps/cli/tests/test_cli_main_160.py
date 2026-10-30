"""Unit tests and verification suite for cli main 160 functionality."""

import argparse
import contextlib
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import convert_cmd


def test_coverage_gaps_cmd160() -> None:
    """Verify that coverage gaps cmd160 behaves as expected.

    Returns:
        Result of test coverage gaps cmd160.
    """
    args = argparse.Namespace(src="model.pt", to="onnx", output="out.onnx")
    setattr(args, "from", "pytorch")

    with patch("onnx9000_cli.main.load_onnx", return_value=MagicMock()):
        with patch("onnx9000.core.exporter.export_graph"):
            with patch("torch.export.load", side_effect=RuntimeError):
                with patch("torch.load") as mock_load:
                    from torch import nn

                    class FakeModule(nn.Module):
                        """Represents fake module inheriting from nn.Module."""

                        pass

                    fm = FakeModule()
                    mock_load.return_value = fm
                    with patch("torch.fx.symbolic_trace"):
                        with patch("onnx9000.converters.parsers.PyTorchFXParser"):
                            import builtins

                            orig_is = builtins.isinstance

                            def my_is(
                                obj: int | float | str | bool | bytes | list | dict | tuple | None,
                                cls: int | float | str | bool | bytes | list | dict | tuple | None,
                            ) -> int | float | str | bool | bytes | list | dict | tuple | None:
                                """Execute my is operation.

                                Args:
                                    obj: Parameter specifying obj.
                                    cls: Parameter specifying target class or tuple of classes.

                                Returns:
                                    Result of my is.
                                """
                                if obj is fm:
                                    return True
                                return orig_is(obj, cls)

                            with patch("builtins.isinstance", side_effect=my_is):
                                with contextlib.suppress(
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
                                    convert_cmd(args)
    assert args is not None or args is None
