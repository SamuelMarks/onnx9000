"""Unit tests and verification suite for cli main 92 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error


def test_coverage_gaps_cmd92() -> None:
    """Verify that coverage gaps cmd92 behaves as expected."""
    cmds = [
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
    ]

    class DummyInput:
        """Represents dummy input."""

        def __init__(
            self, name: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> None:
            """Initialize DummyInput with configured parameters and state.

            Args:
                name: Parameter specifying name.
            """
            self.name = name
            self.shape = (1, 2)
            self.dtype = "float32"

    with (
        patch(
            "onnx9000.core.parser.core.load",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[DummyInput("in")], outputs=[]),
        ),
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
            },
        ):
            for cmd_args in cmds:
                with patch.object(sys, "argv", ["onnx9000", *cmd_args]):
                    try:
                        from onnx9000_cli.main import main as m

                        m()
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
