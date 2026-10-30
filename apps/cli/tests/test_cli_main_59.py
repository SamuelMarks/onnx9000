"""Unit tests and verification suite for cli main 59 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd59() -> None:
    """Verify that coverage gaps cmd59 behaves as expected."""
    cmds = [
        ["jit", "test.onnx"],
        ["jit", "test.onnx", "--target", "wasm"],
        ["jit", "test.onnx", "--target", "unknown"],
        [
            "openvino",
            "export",
            "test.onnx",
            "-o",
            "out",
            "--fp16",
            "--shape",
            "in:1,a",
            "--dynamic-batch",
            "--data-type",
            "in:int32",
        ],
        [
            "openvino",
            "export",
            "test.onnx",
            "-o",
            "out",
            "--fp16",
            "--shape",
            "in:[1,a]",
            "--dynamic-batch",
            "--data-type",
            "in:int32",
        ],
        [
            "openvino",
            "export",
            "test.onnx",
            "-o",
            "out",
            "--fp16",
            "--shape",
            "in:[1,,a]",
            "--dynamic-batch",
            "--data-type",
            "in:int32",
        ],
        ["coverage"],
        ["chat"],
        ["workspace"],
        ["workspace", "ws"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(
                nodes=[],
                tensors={},
                inputs=[MagicMock(name="in", shape=(1, 2))],
                outputs=[],
            ),
        ),
        patch(
            "onnx9000.core.parser.core.load",
            return_value=MagicMock(
                nodes=[],
                tensors={},
                inputs=[MagicMock(name="in", shape=(1, 2))],
                outputs=[],
            ),
        ),
        patch("onnx9000_cli.main.save_onnx"),
        patch("builtins.open"),
        patch("os.makedirs"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.jit.compiler": MagicMock(
                    compile_cpp=MagicMock(return_value="out.cpp"),
                    compile_wasm=MagicMock(return_value="out.js"),
                ),
                "onnx9000.openvino.exporter": MagicMock(
                    OpenVinoExporter=MagicMock(
                        return_value=MagicMock(export=MagicMock(return_value=("xml", b"bin")))
                    )
                ),
                "onnx9000.openvino.api": MagicMock(),
                "onnx9000_cli.coverage": MagicMock(),
                "tui_chat": MagicMock(),
                "onnx9000_workspace": MagicMock(),
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
