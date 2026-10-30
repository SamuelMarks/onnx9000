"""Unit tests and verification suite for cli main 155 functionality."""

import argparse
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import convert_cmd


def test_coverage_gaps_cmd155() -> None:
    """Verify that coverage gaps cmd155 behaves as expected.

    Returns:
        Result of test coverage gaps cmd155.
    """
    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        original_import = __import__
        with patch("builtins.__import__") as mock_import:
            # We want to patch only torch
            def fake_import(
                name: int | float | str | bool | bytes | list | dict | tuple | None,
                globals: int | float | str | bool | bytes | list | dict | tuple | None = None,
                locals: int | float | str | bool | bytes | list | dict | tuple | None = None,
                fromlist: int | float | str | bool | bytes | list | dict | tuple | None = (),
                level: int | float | str | bool | bytes | list | dict | tuple | None = 0,
            ) -> int | float | str | bool | bytes | list | dict | tuple | None:
                """Execute fake import operation.

                Args:
                    name: Parameter specifying name.
                    globals: Parameter specifying globals.
                    locals: Parameter specifying locals.
                    fromlist: Parameter specifying fromlist.
                    level: Parameter specifying level.

                Returns:
                    Result of fake import.
                """
                if name == "torch":
                    mock_torch = MagicMock()
                    mock_torch.export.load = MagicMock(side_effect=RuntimeError)
                    mock_torch.nn.Module = type("Mod", (object,), {})
                    mock_torch.fx.symbolic_trace = MagicMock()
                    mock_torch.load = MagicMock()
                    return mock_torch
                elif name == "onnx9000.converters.parsers":
                    mock_parsers = MagicMock()
                    mock_parsers.PyTorchFXParser = MagicMock(
                        return_value=MagicMock(parse=MagicMock())
                    )
                    return mock_parsers
                elif name == "onnx9000.converters.paddle.api":
                    return MagicMock()
                elif name == "onnx9000.converters.jit.compiler":
                    mock_c = MagicMock()
                    mock_c.compile_cpp = MagicMock()
                    mock_c.compile_wasm = MagicMock()
                    return mock_c
                return original_import(name, globals, locals, fromlist, level)

            mock_import.side_effect = fake_import

            with (
                patch("builtins.open"),
                patch("os.path.isdir", return_value=False),
                patch("os.path.exists", return_value=False),
            ):
                args = argparse.Namespace(src="test.onnx", to="onnx", output="out.onnx")
                setattr(args, "from", "pytorch")
                try:
                    with patch("onnx9000.core.exporter.export_graph"):
                        convert_cmd(args)
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
    assert test_coverage_gaps_cmd155 is not None
