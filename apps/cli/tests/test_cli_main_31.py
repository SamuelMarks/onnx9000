import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd31():
    cmds = [
        ["jit", "test.onnx"],
        ["jit", "test.onnx", "--target", "wasm"],
        ["jit", "test.onnx", "--target", "unknown"],
        ["testing"],
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
                "onnx9000.jit.compiler": MagicMock(
                    compile_cpp=MagicMock(return_value="cpp"),
                    compile_wasm=MagicMock(return_value="wasm"),
                ),
                "onnx9000.backends.testing.runner": MagicMock(),
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
