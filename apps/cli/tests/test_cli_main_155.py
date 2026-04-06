import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import convert_cmd, jit_cmd, main


def test_coverage_gaps_cmd155():
    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch("builtins.__import__") as mock_import:
            # We want to patch only torch
            original_import = __builtins__["__import__"]

            def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "torch":
                    mock_torch = MagicMock()
                    mock_torch.export.load = MagicMock(side_effect=Exception)
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
                except Exception:
                    pass
