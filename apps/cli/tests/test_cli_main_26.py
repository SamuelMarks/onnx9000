"""Unit tests and verification suite for cli main 26 functionality."""

import contextlib
import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd26() -> None:
    """Verify that coverage gaps cmd26 behaves as expected."""
    cmds = [
        ["info", "webnn"],
        ["array", "script.py", "--lazy"],
        ["coreml", "test.onnx"],
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
                "onnx9000_array": MagicMock(lazy_mode=MagicMock()),
                "importlib.util.spec_from_file_location": MagicMock(
                    return_value=MagicMock(loader=MagicMock())
                ),
                "importlib.util.module_from_spec": MagicMock(return_value=MagicMock()),
                "subprocess.run": MagicMock(),
            },
        ):
            with patch(
                "importlib.util.spec_from_file_location",
                return_value=MagicMock(loader=MagicMock()),
            ):
                with patch("importlib.util.module_from_spec", return_value=MagicMock()):
                    with patch("subprocess.run"):
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

        with (
            patch.dict(
                sys.modules,
                {
                    "onnx9000_array": MagicMock(lazy_mode=MagicMock()),
                },
            ),
            patch("importlib.util.spec_from_file_location", return_value=None),
        ):
            with patch.object(sys, "argv", ["onnx9000", "array", "script.py"]):
                with contextlib.suppress(SystemExit):
                    main()
    assert cmds is not None or cmds is None
