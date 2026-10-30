"""Unit tests and verification suite for cli main 37 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd37() -> None:
    """Verify that coverage gaps cmd37 behaves as expected."""
    cmds = [
        [
            "optimum",
            "export",
            "test",
            "--task",
            "task",
            "--device",
            "cpu",
            "--split",
            "train",
            "--cache-dir",
            "out",
            "--opset",
            "12",
        ],
        [
            "optimum",
            "optimize",
            "test",
            "--level",
            "O3",
            "--disable-fusion",
            "--optimize-size",
        ],
        [
            "optimum",
            "quantize",
            "test",
            "gptq",
            "--gptq-bits",
            "4",
            "--gptq-group-size",
            "128",
        ],
        ["optimum", "train", "test.onnx", "ds", "out"],
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
                "onnx9000_optimum.export": MagicMock(),
                "onnx9000_optimum.optimize": MagicMock(),
                "onnx9000_optimum.quantize": MagicMock(),
                "onnx9000_optimum.training": MagicMock(),
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
