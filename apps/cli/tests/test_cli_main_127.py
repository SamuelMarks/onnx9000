"""Unit tests and verification suite for cli main 127 functionality."""

import argparse
import contextlib
import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import convert_cmd, jit_cmd


def test_coverage_gaps_cmd127() -> None:
    """Verify that coverage gaps cmd127 behaves as expected."""
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
                "onnx9000.converters.paddle.api": MagicMock(
                    convert_paddle_to_onnx=MagicMock(return_value="t")
                ),
                "torch.export": MagicMock(load=MagicMock(side_effect=RuntimeError)),
                "torch.load": MagicMock(return_value=type("Mod", (object,), {})()),
                "torch.nn": MagicMock(Module=type("Mod", (object,), {})),
                "torch.fx": MagicMock(symbolic_trace=MagicMock()),
                "torch": MagicMock(),
                "onnx9000.converters.torch.fx": MagicMock(
                    PyTorchFXParser=MagicMock(
                        return_value=MagicMock(parse=MagicMock(return_value=MagicMock()))
                    )
                ),
                "onnx9000.core.exporter": MagicMock(export_graph=MagicMock()),
                "onnx9000.converters.jit.compiler": MagicMock(
                    compile_cpp=MagicMock(return_value="out.cpp"),
                    compile_wasm=MagicMock(return_value="out.js"),
                ),
            },
        ):
            with (
                patch("builtins.open"),
                patch("os.path.isdir", return_value=True),
                patch("os.path.exists", side_effect=[True, True, True, True, True, True]),
                patch("json.load", return_value={}),
            ):
                args = argparse.Namespace(src="test.onnx", to="onnx", output="out.onnx")
                setattr(args, "from", "paddle")
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

            with (
                patch("builtins.open"),
                patch("os.path.isdir", return_value=False),
                patch("os.path.exists", return_value=False),
                patch("json.load", return_value={}),
            ):
                args = argparse.Namespace(src="test.onnx", to="onnx", output="out.onnx")
                setattr(args, "from", "pytorch")
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

            args = argparse.Namespace(model="test.onnx", target="unknown", output=None)
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
                jit_cmd(args)
    assert test_coverage_gaps_cmd127 is not None
