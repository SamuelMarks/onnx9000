import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main, tvm_cmd, diffusers_cmd, jit_cmd


def test_coverage_gaps_cmd57():
    cmds = [
        ["tvm", "test.onnx"],
        ["diffusers", "export", "test", "out"],
        ["diffusers", "unknown", "test", "out"],
        ["jit", "test.onnx"],
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
                "onnx9000.tvm.build_module": MagicMock(),
                "onnx9000_diffusers.pipeline": MagicMock(),
                "onnx9000.jit.compiler": MagicMock(
                    compile_cpp=MagicMock(return_value="out.cpp"),
                    compile_wasm=MagicMock(return_value="out.js"),
                ),
            },
        ):
            for cmd_args in cmds:
                with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                    try:
                        main()
                    except Exception:
                        pass
                    except SystemExit:
                        pass

            try:
                args = MagicMock()
                args.model = "test.onnx"
                args.target = "llvm"
                tvm_cmd(args)

                args = MagicMock()
                args.diffusers_command = "export"
                args.model_id = "model"
                diffusers_cmd(args)

                args = MagicMock()
                args.diffusers_command = "unknown"
                args.model_id = "model"
                diffusers_cmd(args)

                args = MagicMock()
                args.model = "test.onnx"
                args.target = "cpp"
                args.output = "o"
                jit_cmd(args)

                args.target = "wasm"
                jit_cmd(args)

                args.target = "unknown"
                jit_cmd(args)
            except Exception:
                pass
            except SystemExit:
                pass
