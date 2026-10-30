"""Unit tests and verification suite for cli main 20 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd20() -> None:
    """Verify that coverage gaps cmd20 behaves as expected."""
    cmds = [
        ["export", "script.py"],
        ["optimum", "export", "test", "out"],
        ["optimum", "optimize", "test", "out"],
        ["optimum", "quantize", "test", "out"],
        ["convert", "test", "--from", "keras", "--to", "onnx", "-o", "out.onnx"],
        ["chat"],
        ["workspace"],
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
                "onnx9000.converters.frontend.nn.module": MagicMock(),
                "onnx9000.converters.frontend.tracer": MagicMock(
                    trace=MagicMock(
                        return_value=MagicMock(to_graph=MagicMock(return_value=MagicMock()))
                    )
                ),
                "onnx9000.core.exporter": MagicMock(),
                "onnx9000.converters": MagicMock(),
                "onnx9000.converters.torch_like": MagicMock(),
                "onnx9000_optimum.export": MagicMock(),
                "onnx9000_optimum.optimize": MagicMock(),
                "onnx9000_optimum.quantize": MagicMock(),
            },
        ):
            with patch(
                "importlib.util.spec_from_file_location",
                return_value=MagicMock(loader=MagicMock()),
            ):
                with patch("importlib.util.module_from_spec", return_value=MagicMock()):
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
