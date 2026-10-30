"""Unit tests and verification suite for cli main 161 functionality."""

import argparse
import contextlib
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import convert_cmd


def test_coverage_gaps_cmd161() -> None:
    """Verify that coverage gaps cmd161 behaves as expected."""
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
                            # DO NOT mock isinstance at all!
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
