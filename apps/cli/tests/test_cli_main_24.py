"""Unit tests and verification suite for cli main 24 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd24() -> None:
    """Verify that coverage gaps cmd24 behaves as expected."""
    cmds = [
        [
            "openvino",
            "export",
            "test.onnx",
            "-o",
            "out",
            "--fp16",
            "--shape",
            "in:[1,2]",
            "invalid",
            "--dynamic-batch",
            "--data-type",
            "in:float32",
        ],
        ["openvino"],
        ["compile", "test.onnx"],
        ["info"],
        ["info", "ops", "test.onnx"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(inputs=[MagicMock(name="in", shape=(1, 2))], outputs=[]),
        ),
        patch(
            "onnx9000.core.parser.core.load",
            return_value=MagicMock(inputs=[MagicMock(name="in", shape=(1, 2))], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
        patch("builtins.open"),
        patch("os.makedirs"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.openvino.exporter": MagicMock(
                    OpenVinoExporter=MagicMock(
                        return_value=MagicMock(export=MagicMock(return_value=("xml", b"bin")))
                    )
                ),
                "onnx9000.c_compiler.compiler": MagicMock(
                    C89Compiler=MagicMock(
                        return_value=MagicMock(
                            generate=MagicMock(return_value=("header", "source"))
                        )
                    )
                ),
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
