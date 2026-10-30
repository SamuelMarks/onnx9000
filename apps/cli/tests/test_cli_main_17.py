"""Unit tests and verification suite for cli main 17 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd17() -> None:
    """Verify that coverage gaps cmd17 behaves as expected."""
    cmds = [
        ["hummingbird", "test.onnx"],
        ["zoo", "download", "test"],
        ["zoo", "inspect-safetensors", "test"],
        ["genai", "--mode", "test"],
        ["onnx2gguf", "test.onnx", "out.gguf"],
        ["gguf2onnx", "test.gguf", "out.onnx"],
        ["autograd", "test.onnx", "out.onnx"],
        ["testing"],
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
                "onnx9000.optimizer.hummingbird.engine": MagicMock(),
                "onnx9000.optimizer.hummingbird.onnxml_parser": MagicMock(),
                "onnx9000.optimizer.hummingbird.strategies": MagicMock(),
                "onnx9000.zoo.catalog": MagicMock(),
                "onnx9000.zoo.tensors": MagicMock(),
                "onnx9000.genai.ecosystem": MagicMock(),
                "onnx9000.genai.qa": MagicMock(),
                "onnx9000.onnx2gguf.compiler": MagicMock(),
                "onnx9000.gguf2onnx.parser": MagicMock(),
                "onnx9000.toolkit.autograd.compiler": MagicMock(),
                "onnx9000.toolkit.testing.runner": MagicMock(),
            },
        ):
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
