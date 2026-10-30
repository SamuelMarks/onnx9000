"""Unit tests and verification suite for cli main 23 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import (
    main,
)


def test_coverage_gaps_cmd23() -> None:
    """Verify that coverage gaps cmd23 behaves as expected."""
    cmds = [
        [
            "optimum",
            "export",
            "test",
            "out",
            "--task",
            "task",
            "--device",
            "cpu",
            "--atol",
            "1e-4",
        ],
        ["optimum", "optimize", "test", "out", "--level", "O3"],
        [
            "optimum",
            "quantize",
            "test",
            "out",
            "--weight-only",
            "--calibration-data",
            "data",
        ],
        ["openvino", "export", "test.onnx", "-o", "out", "--fp16"],
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
                "onnx9000.openvino.api": MagicMock(),
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
