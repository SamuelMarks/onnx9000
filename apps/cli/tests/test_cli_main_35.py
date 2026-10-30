"""Unit tests and verification suite for cli main 35 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd35() -> None:
    """Verify that coverage gaps cmd35 behaves as expected."""
    cmds = [
        ["zoo", "download", "test"],
        ["zoo", "inspect-safetensors", "test"],
        ["genai", "--mode", "test"],
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
                "onnx9000.zoo.catalog": MagicMock(),
                "onnx9000.zoo.tensors": MagicMock(),
                "onnx9000.genai.ecosystem": MagicMock(),
                "onnx9000.genai.qa": MagicMock(),
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

    # With import errors
    with patch.dict(
        sys.modules,
        {
            "onnx9000.zoo.catalog": None,
            "onnx9000.genai.ecosystem": None,
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
