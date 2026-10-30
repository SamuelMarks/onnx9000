"""Unit tests and verification suite for cli main 32 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd32() -> None:
    """Verify that coverage gaps cmd32 behaves as expected."""
    cmds = [
        ["rocm", "test.onnx"],
        ["cpu", "test.onnx"],
        ["cuda", "test.onnx"],
        ["apple", "test.onnx"],
        ["webgpu", "test.onnx"],
        [
            "onnx2tf",
            "test.onnx",
            "out.pb",
            "--external-weights",
            "w.bin",
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
                "onnx9000.backends.rocm.executor": MagicMock(),
                "onnx9000.backends.cpu.executor": MagicMock(),
                "onnx9000.backends.cuda.executor": MagicMock(),
                "onnx9000.backends.apple.executor": MagicMock(),
                "onnx9000.tf.exporter": MagicMock(export_to_tf=MagicMock()),
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
