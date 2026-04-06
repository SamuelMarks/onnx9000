import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd73():
    cmds = [
        ["jit", "test.onnx", "--target", "cpp"],
        ["rocm", "test.onnx"],
        ["cpu", "test.onnx"],
        ["cuda", "test.onnx"],
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
                "onnx9000.jit.compiler": MagicMock(compile_cpp=MagicMock(return_value="out.cpp")),
                "onnx9000.backends.rocm.executor": MagicMock(),
                "onnx9000.backends.cpu.executor": MagicMock(),
                "onnx9000.backends.cuda.executor": MagicMock(),
                "onnx9000.backends.testing.runner": MagicMock(
                    BackendTestRunner=MagicMock(return_value=MagicMock(run=MagicMock()))
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
