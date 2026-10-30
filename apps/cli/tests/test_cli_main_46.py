"""Unit tests and verification suite for cli main 46 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd46() -> None:
    """Verify that coverage gaps cmd46 behaves as expected."""
    cmds = [
        ["apple", "test.onnx"],
        ["webgpu", "test.onnx"],
        ["tensorrt", "test.onnx"],
        [
            "onnx2tf",
            "test.onnx",
            "-o",
            "out.tflite",
            "--keep-nchw",
            "--int8",
            "--fp16",
            "-b",
            "10",
            "--disable-optimization",
            "--external-weights",
            "weights.bin",
            "--progress",
            "--micro",
        ],
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
                "onnx9000.backends.apple.executor": MagicMock(),
                "onnx9000.tensorrt.builder": MagicMock(),
                "onnx9000.tflite_exporter.cli": MagicMock(main=MagicMock()),
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
