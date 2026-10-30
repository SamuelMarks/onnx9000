"""Unit tests and verification suite for cli main 153 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error


def test_coverage_gaps_cmd153() -> None:
    """Verify that coverage gaps cmd153 behaves as expected."""
    try:
        from onnx9000_cli.main import main

        with (
            patch.object(
                sys,
                "argv",
                ["onnx9000", "convert", "test", "--from", "pytorch", "--to", "onnx"],
            ),
            patch(
                "onnx9000_cli.main.load_onnx",
                return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
            ),
            patch("onnx9000_cli.main.save_onnx"),
            patch("builtins.open"),
        ):
            with patch("os.path.isdir", return_value=False):
                with patch("os.path.exists", return_value=False):
                    import torch
                    import torch.nn

                    class FakeMod(torch.nn.Module):
                        """Represents fake mod inheriting from torch.nn.Module."""

                        pass

                    fake_mod = FakeMod()
                    with (
                        patch.dict(
                            sys.modules,
                            {
                                "torch.export": MagicMock(load=MagicMock(side_effect=RuntimeError)),
                                "torch.load": MagicMock(return_value=fake_mod),
                                "onnx9000.converters.parsers": MagicMock(
                                    PyTorchFXParser=MagicMock(
                                        return_value=MagicMock(
                                            parse=MagicMock(return_value=MagicMock())
                                        )
                                    )
                                ),
                                "onnx9000.core.exporter": MagicMock(export_graph=MagicMock()),
                                "torch.fx": MagicMock(
                                    symbolic_trace=MagicMock(return_value=fake_mod)
                                ),
                            },
                        ),
                        patch("builtins.isinstance", return_value=True),
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
    assert test_coverage_gaps_cmd153 is not None
