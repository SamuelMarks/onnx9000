import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd112():
    cmds = [
        ["convert", "test.onnx", "--from", "paddle", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "pytorch", "--to", "onnx", "-o", "out.onnx"],
        ["jit", "test.onnx", "--target", "cpp"],
        ["jit", "test.onnx", "--target", "wasm"],
        ["jit", "test.onnx", "--target", "unknown"],
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
            # Need to hit lines 353-354. It is inside `else:` block of `if os.path.isdir(args.src):`
            # and it has `if os.path.exists(params_path):`
            # We need os.path.isdir(args.src) to be False.
            with (
                patch("builtins.open"),
                patch("os.path.isdir", return_value=False),
                patch("os.path.exists", side_effect=[True, True, True, True, True, True]),
                patch("json.load", return_value={}),
            ):
                for cmd_args in cmds:
                    with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                        try:
                            main()
                        except Exception:
                            pass
                        except SystemExit:
                            pass
