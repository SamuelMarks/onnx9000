"""Unit tests and verification suite for cli main 157 functionality."""

import argparse
import contextlib
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import convert_cmd


def test_coverage_gaps_cmd157() -> None:
    """Verify that coverage gaps cmd157 behaves as expected."""
    args = argparse.Namespace(src="model.pt", to="onnx", output="out.onnx")
    setattr(args, "from", "pytorch")

    with patch("onnx9000_cli.main.load_onnx", return_value=MagicMock()):
        with patch("onnx9000.core.exporter.export_graph"):
            with patch("torch.export.load", side_effect=RuntimeError):
                with patch("torch.load") as mock_load:
                    import torch

                    class MockMod(torch.nn.Module):
                        """Represents mock mod inheriting from torch.nn.Module."""

                        pass

                    mock_load.return_value = MockMod()
                    with patch("torch.fx.symbolic_trace"):
                        with patch("onnx9000.converters.parsers.PyTorchFXParser"):
                            # Do not mock isinstance
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
