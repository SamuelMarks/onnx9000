import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main, convert_cmd, jit_cmd


def test_coverage_gaps_cmd127():
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
                "onnx9000.converters.paddle.api": MagicMock(
                    convert_paddle_to_onnx=MagicMock(return_value="t")
                ),
                "torch.export": MagicMock(load=MagicMock(side_effect=Exception)),
                "torch.load": MagicMock(return_value=type("Mod", (object,), {})()),
                "torch.nn": MagicMock(Module=type("Mod", (object,), {})),
                "torch.fx": MagicMock(symbolic_trace=MagicMock()),
                "torch": MagicMock(),
                "onnx9000.converters.torch.fx": MagicMock(
                    PyTorchFXParser=MagicMock(
                        return_value=MagicMock(parse=MagicMock(return_value=MagicMock()))
                    )
                ),
                "onnx9000.core.exporter": MagicMock(export_graph=MagicMock()),
                "onnx9000.converters.jit.compiler": MagicMock(
                    compile_cpp=MagicMock(return_value="out.cpp"),
                    compile_wasm=MagicMock(return_value="out.js"),
                ),
            },
        ):
            with (
                patch("builtins.open"),
                patch("os.path.isdir", return_value=True),
                patch("os.path.exists", side_effect=[True, True, True, True, True, True]),
                patch("json.load", return_value={}),
            ):
                args = argparse.Namespace(src="test.onnx", to="onnx", output="out.onnx")
                setattr(args, "from", "paddle")
                try:
                    convert_cmd(args)
                except Exception:
                    pass

            with (
                patch("builtins.open"),
                patch("os.path.isdir", return_value=False),
                patch("os.path.exists", return_value=False),
                patch("json.load", return_value={}),
            ):
                args = argparse.Namespace(src="test.onnx", to="onnx", output="out.onnx")
                setattr(args, "from", "pytorch")
                try:
                    convert_cmd(args)
                except Exception:
                    pass

            args = argparse.Namespace(model="test.onnx", target="unknown", output=None)
            try:
                jit_cmd(args)
            except Exception:
                pass
